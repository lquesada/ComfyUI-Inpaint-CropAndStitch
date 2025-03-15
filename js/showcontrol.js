import { app } from "../../scripts/app.js";

// Some fragments of this code are from https://github.com/LucianoCirino/efficiency-nodes-comfyui

function inpaintCropAndStitchHandler(node) {
    if (node.comfyClass == "InpaintCrop") {
        toggleWidget(node, findWidgetByName(node, "force_width"));
        toggleWidget(node, findWidgetByName(node, "force_height"));
        toggleWidget(node, findWidgetByName(node, "rescale_factor"));
        toggleWidget(node, findWidgetByName(node, "min_width"));
        toggleWidget(node, findWidgetByName(node, "min_height"));
        toggleWidget(node, findWidgetByName(node, "max_width"));
        toggleWidget(node, findWidgetByName(node, "max_height"));
        toggleWidget(node, findWidgetByName(node, "padding"));
        if (findWidgetByName(node, "mode").value == "free size") {
            toggleWidget(node, findWidgetByName(node, "rescale_factor"), true);
            toggleWidget(node, findWidgetByName(node, "padding"), true);
        }
        else if (findWidgetByName(node, "mode").value == "ranged size") {
            toggleWidget(node, findWidgetByName(node, "min_width"), true);
            toggleWidget(node, findWidgetByName(node, "min_height"), true);
            toggleWidget(node, findWidgetByName(node, "max_width"), true);
            toggleWidget(node, findWidgetByName(node, "max_height"), true);
            toggleWidget(node, findWidgetByName(node, "padding"), true);
        }
        else if (findWidgetByName(node, "mode").value == "forced size") {
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
        if (findWidgetByName(node, "mode").value == "factors") {
            toggleWidget(node, findWidgetByName(node, "expand_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_right_factor"), true);
        }
        if (findWidgetByName(node, "mode").value == "pixels") {
            toggleWidget(node, findWidgetByName(node, "expand_up_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_down_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_left_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_right_pixels"), true);
        }
    } else if (node.comfyClass == "InpaintResize") {
        toggleWidget(node, findWidgetByName(node, "min_width"));
        toggleWidget(node, findWidgetByName(node, "min_height"));
        toggleWidget(node, findWidgetByName(node, "rescale_factor"));
        if (findWidgetByName(node, "mode").value == "ensure minimum size") {
            toggleWidget(node, findWidgetByName(node, "min_width"), true);
            toggleWidget(node, findWidgetByName(node, "min_height"), true);
        }
        else if (findWidgetByName(node, "mode").value == "factor") {
            toggleWidget(node, findWidgetByName(node, "rescale_factor"), true);
        }
    }
    return;
}

let origProps = {};

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

const doesInputWithNameExist = (node, name) => {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
};

const HIDDEN_TAG = "tschide";
// Toggle Widget + change size
function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
            
    // Store the original properties of the widget if not already stored
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }       
        
    const origSize = node.size;

    // Set the widget type and computeSize based on the show flag
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    
    // Recursively handle linked widgets if they exist
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));
        
    // Calculate the new height for the node based on its computeSize method
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}   

app.registerExtension({
    name: "inpaint-cropandstitch.showcontrol",
    nodeCreated(node) {
        if (!node.comfyClass.startsWith("Inpaint")) {
            return;
        }

        inpaintCropAndStitchHandler(node);
        for (const w of node.widgets || []) {
            let widgetValue = w.value;

            // Store the original descriptor if it exists 
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value') || 
                Object.getOwnPropertyDescriptor(Object.getPrototypeOf(w), 'value');

            Object.defineProperty(w, 'value', {
                get() {
                    // If there's an original getter, use it. Otherwise, return widgetValue.
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {
                    // If there's an original setter, use it. Otherwise, set widgetValue.
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else { 
                        widgetValue = newVal;
                    }

                    inpaintCropAndStitchHandler(node);
                }
            });
        }
    }
});
