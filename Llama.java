package the.hs;

public class Llama {
    static {
        System.loadLibrary("llama");
    }

    public native int initEngine(String nativeLibDir, String modelPath);
    public native int startGeneration(String[] roles, String[] contents);
    public native String generateNextToken();
}
