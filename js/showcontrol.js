import { app } from "../../scripts/app.js";

// Some fragments of this code are from https://github.com/LucianoCirino/efficiency-nodes-comfyui

function inpaintCropAndStitchHandler(node) {
    if (node.comfyClass == "InpaintCropImproved") {
        toggleWidget(node, findWidgetByName(node, "preresize_mode"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_height"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_height"));
        const preresize = findWidgetByName(node, "preresize");
        if (preresize && preresize.value == true) {
            toggleWidget(node, findWidgetByName(node, "preresize_mode"), true);
            toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
            toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
            toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
            toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
        }
        toggleWidget(node, findWidgetByName(node, "extend_up_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_down_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_left_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_right_factor"));
        const extendForOutpainting = findWidgetByName(node, "extend_for_outpainting");
        if (extendForOutpainting && extendForOutpainting.value == true) {
            toggleWidget(node, findWidgetByName(node, "extend_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_right_factor"), true);
        }
        toggleWidget(node, findWidgetByName(node, "output_target_width"));
        toggleWidget(node, findWidgetByName(node, "output_target_height"));
        const outputResize = findWidgetByName(node, "output_resize_to_target_size");
        if (outputResize && outputResize.value == true) {
            toggleWidget(node, findWidgetByName(node, "output_target_width"), true);
            toggleWidget(node, findWidgetByName(node, "output_target_height"), true);
        }
    }

    // OLD
    if (node.comfyClass == "InpaintCrop") {
        toggleWidget(node, findWidgetByName(node, "force_width"));
        toggleWidget(node, findWidgetByName(node, "force_height"));
        toggleWidget(node, findWidgetByName(node, "rescale_factor"));
        toggleWidget(node, findWidgetByName(node, "min_width"));
        toggleWidget(node, findWidgetByName(node, "min_height"));
        toggleWidget(node, findWidgetByName(node, "max_width"));
        toggleWidget(node, findWidgetByName(node, "max_height"));
        toggleWidget(node, findWidgetByName(node, "padding"));
        const mode = findWidgetByName(node, "mode");
        if (mode && mode.value == "free size") {
            toggleWidget(node, findWidgetByName(node, "rescale_factor"), true);
            toggleWidget(node, findWidgetByName(node, "padding"), true);
        }
        else if (mode && mode.value == "ranged size") {
            toggleWidget(node, findWidgetByName(node, "min_width"), true);
            toggleWidget(node, findWidgetByName(node, "min_height"), true);
            toggleWidget(node, findWidgetByName(node, "max_width"), true);
            toggleWidget(node, findWidgetByName(node, "max_height"), true);
            toggleWidget(node, findWidgetByName(node, "padding"), true);
        }
        else if (mode && mode.value == "forced size") {
            toggleWidget(node, findWidgetByName(node, "force_width"), true);
            toggleWidget(node, findWidgetByName(node, "force_height"), true);
        }
    } else if (node.comfyClass == "InpaintExtendOutpaint") {
        toggleWidget(node, findWidgetByName(node, "expand_up_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_up_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_down_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_down_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_left_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_left_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_right_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_right_factor"));
        const mode = findWidgetByName(node, "mode");
        if (mode && mode.value == "factors") {
            toggleWidget(node, findWidgetByName(node, "expand_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_right_factor"), true);
        }
        if (mode && mode.value == "pixels") {
            toggleWidget(node, findWidgetByName(node, "expand_up_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_down_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_left_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_right_pixels"), true);
        }
    } else if (node.comfyClass == "InpaintResize") {
        toggleWidget(node, findWidgetByName(node, "min_width"));
        toggleWidget(node, findWidgetByName(node, "min_height"));
        toggleWidget(node, findWidgetByName(node, "rescale_factor"));
        const mode = findWidgetByName(node, "mode");
        if (mode && mode.value == "ensure minimum size") {
            toggleWidget(node, findWidgetByName(node, "min_width"), true);
            toggleWidget(node, findWidgetByName(node, "min_height"), true);
        }
        else if (mode && mode.value == "factor") {
            toggleWidget(node, findWidgetByName(node, "rescale_factor"), true);
        }
    }
    return;
}

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

// Toggle Widget + change size
function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget) return;
    widget.disabled = !show;
    if (widget.options) {
        widget.options.disabled = !show;
    }
    if (widget._state) {
        widget._state.disabled = !show;
    }
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));
}

function getPropertyDescriptor(obj, prop) {
    let current = obj;
    while (current) {
        const desc = Object.getOwnPropertyDescriptor(current, prop);
        if (desc) return desc;
        current = Object.getPrototypeOf(current);
    }
    return null;
}

app.registerExtension({
    name: "inpaint-cropandstitch.showcontrol",
    nodeCreated(node) {
        if (!node.comfyClass || !node.comfyClass.startsWith("Inpaint")) {
            return;
        }

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function() {
            const res = origOnConfigure?.apply(this, arguments);
            inpaintCropAndStitchHandler(this);
            return res;
        };

        for (const w of node.widgets || []) {
            let widgetValue = w.value;
            const originalDescriptor = getPropertyDescriptor(w, 'value');

            Object.defineProperty(w, 'value', {
                get() {
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(this)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(this, newVal);
                    } else { 
                        widgetValue = newVal;
                    }

                    inpaintCropAndStitchHandler(node);
                },
                configurable: true,
                enumerable: true
            });

            const origCallback = w.callback;
            w.callback = function() {
                const res = origCallback?.apply(this, arguments);
                inpaintCropAndStitchHandler(node);
                return res;
            };
        }

        inpaintCropAndStitchHandler(node);
    },
    loadedGraphNode(node) {
        if (node.comfyClass && node.comfyClass.startsWith("Inpaint")) {
            inpaintCropAndStitchHandler(node);
        }
    }
});
