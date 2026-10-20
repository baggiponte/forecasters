export default {
  render({ model, el, signal }) {
    const host = {
      page: false,
      liveUpdates: false,
      preview: false,
      useHash: false,
      models: model.get("model_info") || [],
      loadModel() {
        return Promise.resolve();
      },
      request(payload, { onResult, onError }) {
        const requestId = payload.request_id || payload.seq;
        let cancelled = false;
        let finished = false;
        const off = () => {
          if (finished) return;
          finished = true;
          model.off("msg:custom", listener);
        };
        const handle = (result) => {
          if (cancelled || !result || result.request_id !== requestId) return;
          if (result.error) {
            onError({ error: String(result.error) });
            off();
            return;
          }
          const results = result.results || {};
          for (const key of payload.models) {
            const item = results[key];
            if (!item) {
              onError({ key, error: "No result for this model." });
            } else if (item.error) {
              onError({ key, error: String(item.error).split("\n")[0] });
            } else if (!Array.isArray(item.median)) {
              onError({ key, error: "Invalid median response." });
            } else {
              onResult({
                key,
                median: item.median,
                interval: item.interval,
                ms: item.ms,
              });
            }
          }
          off();
        };
        const listener = (message) => {
          const content = message && message.content ? message.content : message;
          if (!content || content.type !== "lineup_result") return;
          handle(content.result);
        };
        model.on("msg:custom", listener);
        model.send({ type: "lineup_request", request: payload });
        return () => {
          cancelled = true;
          off();
        };
      },
    };

    const app = mountLineup(el, host);
    const onModels = () => app.setModels(model.get("model_info") || []);
    model.on("change:model_info", onModels);
    const cleanup = () => {
      model.off("change:model_info", onModels);
      app.destroy();
    };
    signal.addEventListener("abort", cleanup, { once: true });
    return cleanup;
  },
};
