'use strict';

// ref: https://developers.google.com/web/ilt/pwa/caching-files-with-service-worker
// ref: https://qiita.com/y_fujieda/items/f9e765ac9d89ba241154

const CACHE_NAME = 'main-rev-1';
const URLS_TO_CACHE = [
  './index.html',
  './bundle.js',
  './saved_model_js/tensorflowjs_model.pb',
  './saved_model_js/weights_manifest.json',
  './saved_model_js/group1-shard1of4',
  './saved_model_js/group1-shard2of4',
  './saved_model_js/group1-shard3of4',
  './saved_model_js/group1-shard4of4',
  'https://cdnjs.cloudflare.com/ajax/libs/signature_pad/1.5.3/signature_pad.min.js'
];

self.addEventListener('install', event => {
  console.log('SW: Service Worker installed');
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('SW: Adding to cache:', CACHE_NAME, 'files:', URLS_TO_CACHE);
      return cache.addAll(URLS_TO_CACHE);
    })
  );
});

self.addEventListener('activate', event => {
  console.log('SW: Service Worker became active');
  // delete old cache if exists
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          // delete the cache if its name (revision) is not the current one
          if (cacheName !== CACHE_NAME) {
            console.log('SW: Deleting cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// If a request doesn't match anything in the cache,
// get it from the network, send it to the page
// and add it to the cache at the same time.
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.open(CACHE_NAME).then(cache => {
      return cache.match(event.request).then(response => {
        if (response) {
          console.log('SW: Fetch: read from cache:', CACHE_NAME, 'file:', event.request.url);
          return response;
        }
        fetch(event.request).then(netResponse => {
          console.log('SW: Fetch: read from the network:', event.request.url);
          cache.put(event.request, netResponse.clone());
          return netResponse;
        });
      });
    })
  );
});
