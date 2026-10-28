/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the .onnx model files, e.g. a Hugging Face repository's resolve/main/; unset: models/ next to the page */
  readonly VITE_MODELS_URL?: string;
}
