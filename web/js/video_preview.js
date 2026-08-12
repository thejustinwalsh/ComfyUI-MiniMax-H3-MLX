import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "MiniMaxH3.MLXVideoPreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3SaveVideo") {
            return;
        }

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            originalOnExecuted?.apply(this, arguments);

            const item = message?.minimax_h3_video?.[0];
            if (!item) {
                return;
            }

            if (!this._minimaxH3Video) {
                const video = document.createElement("video");
                video.controls = true;
                video.loop = true;
                video.muted = false;
                video.playsInline = true;
                video.style.width = "100%";
                video.style.display = "block";
                video.style.background = "#111";
                video.style.borderRadius = "4px";

                this._minimaxH3Video = video;
                this.addDOMWidget("minimax_h3_preview", "video", video, {
                    serialize: false,
                    hideOnZoom: false,
                    getValue() {
                        return "";
                    },
                    setValue() {},
                });
            }

            const params = new URLSearchParams({
                filename: item.filename,
                subfolder: item.subfolder || "",
                type: item.type || "output",
                t: Date.now().toString(),
            });
            this._minimaxH3Video.src = api.apiURL(`/view?${params.toString()}`);
            this._minimaxH3Video.load();
            this.setSize([Math.max(this.size[0], 360), Math.max(this.size[1], 420)]);
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
