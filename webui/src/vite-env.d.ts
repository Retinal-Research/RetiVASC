/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the model host, one folder per model, e.g. a Hugging Face repository at a tag (…/resolve/v0.1.0/); unset: models/ next to the page */
  readonly VITE_MODELS_URL?: string;
}
