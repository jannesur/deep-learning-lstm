console.log("Projekt gestartet");

console.log("TensorFlow.js geladen:", tf);
console.log("Plotly geladen:", Plotly);

const predictButton =
    document.getElementById("predict-button");

const nextButton =
    document.getElementById("next-button");

const autoButton =
    document.getElementById("auto-button");

const stopButton =
    document.getElementById("stop-button");

const resetButton =
    document.getElementById("reset-button");

predictButton.addEventListener(
    "click",
    function () {

        console.log("Vorhersage gestartet");

    }
);

nextButton.addEventListener(
    "click",
    function () {

        console.log("Weiter");

    }
);

autoButton.addEventListener(
    "click",
    function () {

        console.log("Auto");

    }
);

stopButton.addEventListener(
    "click",
    function () {

        console.log("Stopp");

    }
);

resetButton.addEventListener(
    "click",
    function () {

        console.log("Reset");

    }
);