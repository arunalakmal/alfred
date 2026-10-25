// Minimal service worker: caches the app shell so the PWA opens instantly on a phone.
// API calls are never cached; they always go to the network.
const SHELL = "alfred-shell-v2";
const ASSETS = ["/", "/static/manifest.webmanifest", "/static/icon.svg", "/static/vendor/chart.umd.min.js"];

self.addEventListener("install", (e) => e.waitUntil(caches.open(SHELL).then((c) => c.addAll(ASSETS))));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== SHELL).map((k) => caches.delete(k))))));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/")) return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
