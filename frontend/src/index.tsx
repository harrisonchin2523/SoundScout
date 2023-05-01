import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import { polyfillCountryFlagEmojis } from "country-flag-emoji-polyfill";

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

polyfillCountryFlagEmojis();
