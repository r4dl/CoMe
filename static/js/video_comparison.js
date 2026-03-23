var comparisonMethod = "MiLo";
var comparisonScenes = ["Barn", "Courthouse", "Meetingroom"];
var selectedComparisonScene = "Barn";
var sceneStates = {};
var stackedMethodOrder = {
    Ours: 0,
    MiLo: 1,
    PGSR: 2
};

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function getStackedSceneVideoPath(scene) {
    return "static/video/" + scene + ".mp4";
}

function updateComparisonMethodButtons() {
    var miloBtn = document.getElementById("btn_method_milo");
    var pgsrBtn = document.getElementById("btn_method_pgsr");
    if (!miloBtn || !pgsrBtn) {
        return;
    }

    if (comparisonMethod === "MiLo") {
        miloBtn.classList.remove("button-17");
        miloBtn.classList.add("button-17-selected");
        pgsrBtn.classList.remove("button-17-selected");
        pgsrBtn.classList.add("button-17");
    } else {
        pgsrBtn.classList.remove("button-17");
        pgsrBtn.classList.add("button-17-selected");
        miloBtn.classList.remove("button-17-selected");
        miloBtn.classList.add("button-17");
    }
}

function updateComparisonSceneButtons() {
    comparisonScenes.forEach(function(scene) {
        var btn = document.getElementById("btn_scene_" + scene.toLowerCase());
        if (!btn) {
            return;
        }
        if (scene === selectedComparisonScene) {
            btn.classList.remove("button-17");
            btn.classList.add("button-17-selected");
        } else {
            btn.classList.remove("button-17-selected");
            btn.classList.add("button-17");
        }
    });
}

function updateVisibleComparisonScene() {
    comparisonScenes.forEach(function(scene) {
        var sceneContainer = document.getElementById(scene.toLowerCase() + "_comparison_scene");
        var sceneState = sceneStates[scene];
        var isActive = scene === selectedComparisonScene;

        if (sceneContainer) {
            sceneContainer.style.display = isActive ? "" : "none";
        }
        if (sceneState) {
            if (isActive) {
                sceneState.activate();
            } else {
                sceneState.deactivate();
            }
        }
    });
}

function setComparisonMethod(method) {
    if (method !== "MiLo" && method !== "PGSR") {
        return;
    }
    if (comparisonMethod === method) {
        return;
    }

    comparisonMethod = method;
    updateComparisonMethodButtons();
}

function setComparisonScene(scene) {
    if (comparisonScenes.indexOf(scene) === -1) {
        return;
    }
    if (selectedComparisonScene === scene) {
        return;
    }

    selectedComparisonScene = scene;
    updateComparisonSceneButtons();
    updateVisibleComparisonScene();
}

function setupSceneComparison(scene) {
    var sceneKey = scene.toLowerCase();
    var canvas = document.getElementById(sceneKey + "_comparison_canvas");
    var stackedVideo = document.getElementById(sceneKey + "_stacked_video");
    if (!canvas || !stackedVideo) {
        return;
    }

    var ctx = canvas.getContext("2d");
    var state = {
        position: 0.5,
        leftButtonDown: false,
        strokeColor: "#FFFFFF44",
        subVidHeight: 0,
        isActive: false,
        rafId: null
    };

    function resizeCanvas() {
        if (!stackedVideo.videoWidth || !stackedVideo.videoHeight) {
            return;
        }
        state.subVidHeight = Math.floor(stackedVideo.videoHeight / 3);
        canvas.width = stackedVideo.videoWidth;
        canvas.height = state.subVidHeight;
    }

    function updatePosition(e) {
        var bcr = canvas.getBoundingClientRect();
        var normalized = (e.clientX - bcr.left) / bcr.width;
        if (Math.abs(normalized - state.position) < 0.1) {
            state.strokeColor = "#FFFFFFAA";
        } else {
            state.strokeColor = "#FFFFFF44";
        }
        if (state.leftButtonDown && Math.abs(normalized - state.position) < 0.5) {
            state.position = clamp(normalized, 0.0, 1.0);
        }
    }

    function drawLoop() {
        state.rafId = null;
        if (!state.isActive) {
            return;
        }

        if (canvas.width > 0 && canvas.height > 0 && stackedVideo.readyState > 2 && state.subVidHeight > 0) {
            var width = canvas.width;
            var height = canvas.height;
            var splitX = clamp(width * state.position, 0.0, width);
            var rightWidth = clamp(width - splitX, 0.0, width);
            var oursRow = stackedMethodOrder.Ours;
            var otherRow = stackedMethodOrder[comparisonMethod];

            ctx.drawImage(stackedVideo, 0, oursRow * state.subVidHeight, width, height, 0, 0, width, height);
            ctx.drawImage(stackedVideo, splitX, otherRow * state.subVidHeight, rightWidth, height, splitX, 0, rightWidth, height);

            ctx.beginPath();
            ctx.moveTo(splitX, 0);
            ctx.lineTo(splitX, height);
            ctx.closePath();
            ctx.strokeStyle = state.strokeColor;
            ctx.lineWidth = 2;
            ctx.stroke();

            var arrowPosY = height / 2;
            var arrowW = height / 70;
            var arrowL = height / 150;
            var arrowOffset = height / 150;

            ctx.beginPath();
            ctx.moveTo(splitX + arrowL + arrowOffset, arrowPosY - arrowW / 2);
            ctx.lineTo(splitX + 2 * arrowL + arrowOffset, arrowPosY);
            ctx.lineTo(splitX + arrowL + arrowOffset, arrowPosY + arrowW / 2);
            ctx.strokeStyle = state.strokeColor;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(splitX - arrowL - arrowOffset, arrowPosY - arrowW / 2);
            ctx.lineTo(splitX - 2 * arrowL - arrowOffset, arrowPosY);
            ctx.lineTo(splitX - arrowL - arrowOffset, arrowPosY + arrowW / 2);
            ctx.strokeStyle = state.strokeColor;
            ctx.stroke();

            var insetHeight = Math.max(24, height * 0.08);
            var insetPadX = Math.max(8, width * 0.015);
            var insetPadY = Math.max(8, height * 0.02);
            ctx.font = "600 " + Math.max(12, Math.floor(height * 0.04)) + "px Jost, sans-serif";
            ctx.textBaseline = "middle";

            var leftText = "Ours";
            var rightText = comparisonMethod;
            var leftInsetWidth = ctx.measureText(leftText).width + 20;
            var rightInsetWidth = ctx.measureText(rightText).width + 20;
            var rightInsetX = width - insetPadX - rightInsetWidth;

            // Clip labels to their visible side so they disappear when covered by the slider.
            if (splitX > 0) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, splitX, height);
                ctx.clip();
                ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
                ctx.fillRect(insetPadX, insetPadY, leftInsetWidth, insetHeight);
                ctx.fillStyle = "#FFFFFF";
                ctx.fillText(leftText, insetPadX + 10, insetPadY + insetHeight / 2);
                ctx.restore();
            }

            if (splitX < width) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(splitX, 0, width - splitX, height);
                ctx.clip();
                ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
                ctx.fillRect(rightInsetX, insetPadY, rightInsetWidth, insetHeight);
                ctx.fillStyle = "#FFFFFF";
                ctx.fillText(rightText, rightInsetX + 10, insetPadY + insetHeight / 2);
                ctx.restore();
            }
        }

        state.rafId = requestAnimationFrame(drawLoop);
    }

    function activateScene() {
        if (state.isActive) {
            return;
        }
        state.isActive = true;
        stackedVideo.play().catch(function() {
            // Ignore autoplay failures; user interaction can start playback later.
        });
        if (state.rafId === null) {
            state.rafId = requestAnimationFrame(drawLoop);
        }
    }

    function deactivateScene() {
        state.isActive = false;
        stackedVideo.pause();
        if (state.rafId !== null) {
            cancelAnimationFrame(state.rafId);
            state.rafId = null;
        }
    }

    canvas.addEventListener("mousemove", updatePosition, false);
    canvas.addEventListener("mousedown", function(e) {
        if (e.which === 1) {
            state.leftButtonDown = true;
            updatePosition(e);
        }
    }, false);
    canvas.addEventListener("mouseup", function(e) {
        if (e.which === 1) {
            state.leftButtonDown = false;
        }
    }, false);
    canvas.addEventListener("mouseleave", function() {
        state.leftButtonDown = false;
        state.strokeColor = "#FFFFFF44";
    }, false);

    stackedVideo.addEventListener("loadedmetadata", resizeCanvas);
    stackedVideo.src = getStackedSceneVideoPath(scene);
    stackedVideo.load();
    sceneStates[scene] = {
        activate: activateScene,
        deactivate: deactivateScene
    };
}

function initMultiSceneComparisons() {
    comparisonScenes.forEach(function(scene) {
        setupSceneComparison(scene);
    });
    updateComparisonMethodButtons();
    updateComparisonSceneButtons();
    updateVisibleComparisonScene();
}
