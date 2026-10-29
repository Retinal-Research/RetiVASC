/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the model host, one folder per model, e.g. a Hugging Face repository at a tag (…/resolve/v0.1.0/); unset: models/ next to the page */
  readonly VITE_MODELS_URL?: string;
  /** Base URL of the VascX models (e.g. Eyened/vascx on Hugging Face at a revision); unset: models/vascx/ next to the page */
  readonly VITE_VASCX_URL?: string;
  /** Base URL of the TPOT enhancement models; unset: models/tpot/ next to the page */
  readonly VITE_TPOT_URL?: string;
}
