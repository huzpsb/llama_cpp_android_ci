constexpr int BATCH_SIZE = 512;
constexpr int MAX_NEW_TOKENS = 1000;
constexpr int TOTAL_CONTEXT_SIZE = 16000;
constexpr float TOP_P = 0.9;
constexpr float TEMPERATURE = 0.5;

// ----------------------------------------------------------------------------------------------------

#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <sstream>
// #define HS_DEBUG


#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <android/log.h>

#define LOG_TAG "LlamaEngine"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static void aichat_android_log_callback(ggml_log_level level, const char* text, void* user_data)
{
    (void)user_data;
    if (level == GGML_LOG_LEVEL_ERROR)
        LOGe("%s", text);
    else if (level == GGML_LOG_LEVEL_WARN)
        LOGi("WARN: %s", text);
    else
        LOGi("%s", text);
}

static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static llama_batch g_batch;
static common_chat_templates_ptr g_chat_templates;
static common_sampler* g_sampler = nullptr;

struct SlotState
{
    int seq_id;
    uint64_t last_used;
    std::vector<llama_token> tokens;
};

static SlotState g_slots[2] = {
    {0, 0, {}},
    {1, 0, {}}
};
static int g_current_slot = 0;
static uint64_t g_time_counter = 0;
static int g_generated_tokens = 0;
static std::string g_cached_token_chars;

static bool is_valid_utf8(const char* string)
{
    if (!string) return true;
    const auto* bytes = (const unsigned char*)string;
    int num;
    while (*bytes != 0x00)
    {
        if ((*bytes & 0x80) == 0x00) num = 1;
        else if ((*bytes & 0xE0) == 0xC0) num = 2;
        else if ((*bytes & 0xF0) == 0xE0) num = 3;
        else if ((*bytes & 0xF8) == 0xF0) num = 4;
        else return false;

        bytes += 1;
        for (int i = 1; i < num; ++i)
        {
            if ((*bytes & 0xC0) != 0x80) return false;
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jint JNICALL
Java_the_hs_Llama_initEngine(JNIEnv* env, jobject, jstring nativeLibDir, jstring jmodel_path)
{
    llama_log_set(aichat_android_log_callback, nullptr);
    const auto* path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);
    llama_backend_init();
    const auto* model_path = env->GetStringUTFChars(jmodel_path, 0);
    llama_model_params model_params = llama_model_default_params();
    g_model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!g_model) return 1;
    const int n_threads = std::max(2, (int)sysconf(_SC_NPROCESSORS_ONLN) - 2);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = TOTAL_CONTEXT_SIZE;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.n_seq_max = 2;

    g_context = llama_init_from_model(g_model, ctx_params);
    if (!g_context) return 2;
    g_batch = llama_batch_init(BATCH_SIZE, 0, 2);
    g_chat_templates = common_chat_templates_init(g_model, "");
    common_params_sampling sparams;
    sparams.temp = TEMPERATURE;
    sparams.top_p = TOP_P;
    g_sampler = common_sampler_init(g_model, sparams);
    LOGi("Engine initialized successfully. 2-Slot KV cache ready.");
    return 0;
}

#ifdef HS_DEBUG
static std::string tokens_to_string(const std::vector<llama_token>& tokens, int max_tokens = 10)
{
    if (tokens.empty()) return "[]";
    std::ostringstream oss;
    oss << "[";
    int count = std::min((int)tokens.size(), max_tokens);
    for (int i = 0; i < count; ++i)
    {
        if (i > 0) oss << ", ";
        std::string piece = common_token_to_piece(g_context, tokens[i]);
        // 转义换行符，避免日志混乱
        for (auto& c : piece)
        {
            if (c == '\n') c = ' ';
            else if (c == '\r') c = ' ';
        }
        oss << piece;
    }
    if (tokens.size() > max_tokens) oss << " ... (" << tokens.size() << " total)";
    else oss << "]";
    return oss.str();
}


static void log_slot_state(const SlotState& slot, int idx)
{
    LOGi("Slot %d (seq_id=%d, last_used=%llu): %zu tokens -> %s",
         idx, slot.seq_id, (unsigned long long)slot.last_used,
         slot.tokens.size(), tokens_to_string(slot.tokens).c_str());
}
#endif


extern "C"
JNIEXPORT jint JNICALL
Java_the_hs_Llama_startGeneration(
    JNIEnv* env, jobject, jobjectArray jroles, jobjectArray jcontents)
{
    g_cached_token_chars.clear();
    const int msg_count = env->GetArrayLength(jroles);


    std::vector<common_chat_msg> history;
    std::string full_prompt = "";
#ifdef HS_DEBUG
    LOGi("======= New Generation Request =======");
    LOGi("Received %d messages:", msg_count);
#endif

    for (int i = 0; i < msg_count; ++i)
    {
        jstring jrole = (jstring)env->GetObjectArrayElement(jroles, i);
        jstring jcontent = (jstring)env->GetObjectArrayElement(jcontents, i);
        const char* role_str = env->GetStringUTFChars(jrole, 0);
        const char* content_str = env->GetStringUTFChars(jcontent, 0);

        common_chat_msg msg;
        msg.role = role_str;
        msg.content = content_str;
        bool is_last = (i == msg_count - 1);
        full_prompt += common_chat_format_single(g_chat_templates.get(), history, msg, is_last && msg.role == "user",
                                                 false);
        history.push_back(msg);

#ifdef HS_DEBUG
        std::string content_preview = content_str;
        if (content_preview.length() > 80) content_preview = content_preview.substr(0, 80) + "...";
        LOGi("  [%d] %s: %s", i, role_str, content_preview.c_str());
#endif

        env->ReleaseStringUTFChars(jrole, role_str);
        env->ReleaseStringUTFChars(jcontent, content_str);
        env->DeleteLocalRef(jrole);
        env->DeleteLocalRef(jcontent);
    }

    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    std::vector<llama_token> prompt_tokens = common_tokenize(g_context, full_prompt, has_chat_template, true);
#ifdef HS_DEBUG
    LOGi("Prompt tokencount: %zu", prompt_tokens.size());
    LOGi("Prompt tokens (first 20): %s", tokens_to_string(prompt_tokens, 20).c_str());
#endif
    if (prompt_tokens.size() + MAX_NEW_TOKENS > TOTAL_CONTEXT_SIZE / 2)
    {
        LOGe("Context overflow: prompt tokens (%zu) + max_new_tokens (%d) exceeds TOTAL_CONTEXT_SIZE / 2 (%d)",
             prompt_tokens.size(), MAX_NEW_TOKENS, TOTAL_CONTEXT_SIZE / 2);
        return -1;
    }

#ifdef HS_DEBUG
    LOGi("---- Slots before choice ----");
    for (int i = 0; i < 2; i++)
    {
        log_slot_state(g_slots[i], i);
    }
#endif

    int best_slot_idx = -1;
    int best_match_len = 0;
    int best_score = -999999;
    for (int i = 0; i < 2; i++)
    {
        int match_len = 0;
        int min_len = std::min((int)prompt_tokens.size(), (int)g_slots[i].tokens.size());
        while (match_len < min_len && prompt_tokens[match_len] == g_slots[i].tokens[match_len])
        {
            match_len++;
        }
        int discarded = (int)g_slots[i].tokens.size() - match_len;
        int score = (match_len * (discarded == 0 ? 4 : 2)) - discarded;
        if (score > best_score)
        {
            best_score = score;
            best_slot_idx = i;
            best_match_len = match_len;
        }
        else if (score == best_score)
        {
            if (best_slot_idx == -1 || g_slots[i].last_used < g_slots[best_slot_idx].last_used)
            {
                best_slot_idx = i;
                best_match_len = match_len;
            }
        }
    }
    int max_prefix_len = best_match_len;

    if (max_prefix_len == 0)
    {
        best_slot_idx = (g_slots[0].last_used < g_slots[1].last_used) ? 0 : 1;
    }
    g_current_slot = best_slot_idx;
    SlotState& slot = g_slots[g_current_slot];
    slot.last_used = ++g_time_counter;

#ifdef HS_DEBUG
    LOGi("Chosen Slot %d, initial max_prefix_len=%d", g_current_slot, max_prefix_len);
#endif

    int n_s = max_prefix_len;
    int n_prompt = prompt_tokens.size();
    int n_cache = slot.tokens.size();
    if (n_s < n_prompt && n_cache > n_s)
    {
        const int min_match = std::min(16, n_prompt - n_s);
        for (int i = n_s + 1; i <= n_cache - min_match; ++i)
        {
            bool found_c = true;
            for (int j = 0; j < min_match; ++j)
            {
                if (slot.tokens[i + j] != prompt_tokens[n_s + j])
                {
                    found_c = false;
                    break;
                }
            }
            if (found_c)
            {
                int gap_len = i - n_s;
#ifdef HS_DEBUG
                LOGi("Detect gap! Shifting 'c' (pos %d) to 's' end (pos %d). Gap size: %d", i, n_s, gap_len);
#endif
                llama_memory_seq_rm(llama_get_memory(g_context), slot.seq_id, n_s, i);
                llama_memory_seq_add(llama_get_memory(g_context), slot.seq_id, i, -1, -gap_len);
                slot.tokens.erase(slot.tokens.begin() + n_s, slot.tokens.begin() + i);
                n_cache = slot.tokens.size();
                while (max_prefix_len < n_prompt && max_prefix_len < n_cache &&
                    prompt_tokens[max_prefix_len] == slot.tokens[max_prefix_len])
                {
                    max_prefix_len++;
                }
                break;
            }
        }
    }

    if (max_prefix_len == prompt_tokens.size() && max_prefix_len > 0)
    {
        max_prefix_len--;
    }
    if (max_prefix_len < slot.tokens.size())
    {
        llama_memory_seq_rm(llama_get_memory(g_context), slot.seq_id, max_prefix_len, -1);
    }
    slot.tokens.resize(max_prefix_len);

#ifdef HS_DEBUG
    LOGi("After RM and resize, Slot %d now has %zu tokens reused, final prompt pos start at %d",
         g_current_slot, slot.tokens.size(), max_prefix_len);
#endif


    for (int i = max_prefix_len; i < (int)prompt_tokens.size(); i += BATCH_SIZE)
    {
        const int cur_batch_size = std::min((int)prompt_tokens.size() - i, BATCH_SIZE);
        common_batch_clear(g_batch);

        for (int j = 0; j < cur_batch_size; j++)
        {
            const llama_token tid = prompt_tokens[i + j];
            const llama_pos pos = i + j;
            const bool want_logit = (i + j == (int)prompt_tokens.size() - 1);
            common_batch_add(g_batch, tid, pos, {slot.seq_id}, want_logit);
            slot.tokens.push_back(tid);
        }

        if (llama_decode(g_context, g_batch) != 0)
        {
            LOGe("llama_decode() failed during prompt processing");
            return 1;
        }
    }

    common_sampler_reset(g_sampler);
    for (const auto& token : prompt_tokens)
    {
        common_sampler_accept(g_sampler, token, false);
    }

#ifdef HS_DEBUG
    LOGi("---- Slot %d after prompt ingest ----", g_current_slot);
    log_slot_state(g_slots[g_current_slot], g_current_slot);
#endif

    g_generated_tokens = 0;
    return 0;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_the_hs_Llama_generateNextToken(JNIEnv* env, jobject)
{
    if (g_generated_tokens >= MAX_NEW_TOKENS)
    {
        return nullptr;
    }

    SlotState& slot = g_slots[g_current_slot];
    const llama_token new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id))
    {
        return nullptr;
    }

    const llama_pos current_pos = slot.tokens.size();
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_pos, {slot.seq_id}, true);
    if (llama_decode(g_context, g_batch) != 0)
    {
        LOGe("llama_decode() failed during generation");
        return nullptr;
    }
    slot.tokens.push_back(new_token_id);
    g_generated_tokens++;
    std::string new_token_chars = common_token_to_piece(g_context, new_token_id);
    g_cached_token_chars += new_token_chars;
    if (is_valid_utf8(g_cached_token_chars.c_str()))
    {
        jstring result = env->NewStringUTF(g_cached_token_chars.c_str());
        g_cached_token_chars.clear();
        return result;
    }
    return env->NewStringUTF("");
}
