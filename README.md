# Lib llama.cpp for Android

**Pre-built. As small as 2MB. No NDK/Kotlin bullshit. Weekly updates. Ez-to-use wrappers. Fully open source.**

<img width="448" height="181" alt="banner" src="https://github.com/user-attachments/assets/a3c7d469-c230-4126-bd37-2dd7c41081d4" />

For other architectures or build intervals, feel free to fork.

## Why this?

Most llama.cpp developers assume I'm building some fancy multi-tenant Android chat UI.  
**Darn it, I'm not.**

I don't want to install the NDK, I hate Kotlin, and I don't care about model reloading.  
I load the model **once** and never unload it. I also don't care about build info or credits.

What I really need:
- Memory efficient
- High compatibility
- Reasonably fast
- Context inspecting support

That's exactly why you might prefer this wrapper over the official bloated builds:

- **~10 tokens/s** prompt preprocessing & **~5 tokens/s** with Qwen3-4B-Instruct-2507 (Q4K_M) generation on OnePlus Turbo 6V (~$200)
- Works on most `arm64-v8a` phones
- Single `.so` file: **4MB** (2MB after compression)
- Straightforward Java API
- No useless stuff
- **100x better KV cache** than the official crap

### Here's how my KV cache works (and why you need it)

You **do not** have to use the same prefix for it to work.

Let's say you have this chat:

```text
System: You are a helpful assistant
User: 1+1=?
Assistant: 1+1=2
User: What's the capital of France?
Assistant: Paris
```

Now you change it to:

```text
System: You are a helpful assistant
User: What's the capital of France?
Assistant: Paris
User: How can I travel there?
```

My KV cache does all the dirty work for you: it shifts the RoPE, deletes the old context, and serves the new answer as if nothing happened.

[See the implementation](https://github.com/huzpsb/llama_cpp_android_ci/blob/d569ff8ad7e71b83862bbfa17568983243259fb5/library.cpp#L259-L294)

## How to use

1. Copy the `.so` file to your `jniLibs` folder.
2. Copy this class into your project:

```java
package the.hs;

public class Llama {
    static {
        System.loadLibrary("llama");
    }

    public native int initEngine(String nativeLibDir, String modelPath);
    public native int startGeneration(String[] roles, String[] contents);
    public native String generateNextToken();
}
```

3. Use it like this (single-thread only!):

```java
Llama llama = new Llama();
llama.initEngine(getApplicationInfo().nativeLibraryDir, "/path/to/model.gguf");

String[] roles = {"system", "user"};
String[] contents = {"You are an assistant.", "Hello!"};
llama.startGeneration(roles, contents);

while (true) {
    String token = llama.generateNextToken();
    if (token == null) break;
}
```

Want to abort generation? Just **stop calling** `generateNextToken()`.  
The `Llama` object stays completely usable — as if you never aborted.

**Stop using the bloated official llama.cpp Android wrapper.**
