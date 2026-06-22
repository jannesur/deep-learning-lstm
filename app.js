console.log("Projekt gestartet");

console.log("TensorFlow.js geladen:", tf);

console.log("Plotly geladen:", Plotly);

let dictionary = [];

let wordToIndex = {};

let indexToWord = {};

let model;

let lossHistory = [];

let lastPredictions = [];

let autoRunning = false;

const sequenceLength = 3;


// Text bereinigen

function cleanText(text) {

    return text
        .toLowerCase()
        .replace(/[.,!?;:()"]/g, "")
        .replace(/\s+/g, " ")
        .trim();
}


// Wörter erzeugen

function splitIntoWords(text) {

    const clean = cleanText(text);

    if (clean.length === 0) {

        return [];
    }

    return clean.split(" ");
}


// Dictionary erzeugen

function createDictionary(words) {

    dictionary = [];

    for (const word of words) {

        if (dictionary.includes(word) === false) {

            dictionary.push(word);
        }
    }

    wordToIndex = {};

    indexToWord = {};

    for (let i = 0; i < dictionary.length; i++) {

        const word = dictionary[i];

        wordToIndex[word] = i;

        indexToWord[i] = word;
    }
}


// One-Hot-Vektor erzeugen

function createOneHotVector(index) {

    const vector = [];

    for (let i = 0; i < dictionary.length; i++) {

        if (i === index) {

            vector.push(1);

        } else {

            vector.push(0);
        }
    }

    return vector;
}


// Trainingsdaten erzeugen

function createTrainingData(words) {

    const inputs = [];

    const labels = [];

    for (let i = 0; i < words.length - sequenceLength; i++) {

        const inputWords =
            words.slice(
                i,
                i + sequenceLength
            );

        const targetWord =
            words[i + sequenceLength];

        const inputSequence = [];

        for (const word of inputWords) {

            const index =
                wordToIndex[word];

            inputSequence.push(
                createOneHotVector(index)
            );
        }

        const targetIndex =
            wordToIndex[targetWord];

        const targetVector =
            createOneHotVector(targetIndex);

        inputs.push(inputSequence);

        labels.push(targetVector);
    }

    return {

        inputs: tf.tensor3d(inputs),

        labels: tf.tensor2d(labels)
    };
}


// Modell erzeugen

function createModel() {

    const newModel = tf.sequential();

    newModel.add(

        tf.layers.lstm({

            units: 100,

            returnSequences: true,

            inputShape: [
                sequenceLength,
                dictionary.length
            ]
        })
    );

    newModel.add(

        tf.layers.lstm({

            units: 100
        })
    );

    newModel.add(

        tf.layers.dense({

            units: dictionary.length,

            activation: "softmax"
        })
    );

    newModel.compile({

        optimizer: tf.train.adam(0.001),

        loss: "categoricalCrossentropy",

        metrics: ["accuracy"]
    });

    return newModel;
}


// Loss plotten

function plotLossHistory() {

    const epochs = [];

    for (let i = 0; i < lossHistory.length; i++) {

        epochs.push(i + 1);
    }

    Plotly.newPlot(

        "loss-plot",

        [
            {
                x: epochs,
                y: lossHistory,
                mode: "lines",
                type: "scatter",
                name: "Loss"
            }
        ],

        {
            title: "Trainings-Loss",

            xaxis: {
                title: "Epoch"
            },

            yaxis: {
                title: "Loss"
            }
        }
    );
}


// Modell trainieren

async function trainModel() {

    const text =
        document
            .getElementById("training-text")
            .value;

    const words =
        splitIntoWords(text);

    if (words.length <= sequenceLength) {

        alert(
            "Der Trainingstext ist zu kurz."
        );

        return;
    }

    createDictionary(words);

    const trainingData =
        createTrainingData(words);

    model = createModel();

    lossHistory = [];

    lastPredictions = [];

    document.getElementById(
        "training-status"
    ).textContent =
        "Training läuft...";

    await model.fit(

        trainingData.inputs,

        trainingData.labels,

        {
            epochs: 20,

            batchSize: 32,

            shuffle: true,

            callbacks: {

                onEpochEnd: function(epoch, logs) {

                    lossHistory.push(
                        logs.loss
                    );
                }
            }
        }
    );

    document.getElementById(
        "training-status"
    ).textContent =
        "Training abgeschlossen.";

    plotLossHistory();

    trainingData.inputs.dispose();

    trainingData.labels.dispose();
}


// Prompt in Tensor umwandeln

function createInputFromPrompt(promptText) {

    const words =
        splitIntoWords(promptText);

    if (words.length < sequenceLength) {

        return null;
    }

    const lastWords =
        words.slice(
            words.length - sequenceLength
        );

    const inputSequence = [];

    for (const word of lastWords) {

        const index =
            wordToIndex[word];

        if (index === undefined) {

            return null;
        }

        inputSequence.push(
            createOneHotVector(index)
        );
    }

    return tf.tensor3d(
        [inputSequence]
    );
}


// Vorhersage berechnen

async function predictNextWords() {

    if (model === undefined) {

        alert(
            "Bitte zuerst trainieren."
        );

        return;
    }

    const promptText =
        document
            .getElementById("prompt-input")
            .value;

    const inputTensor =
        createInputFromPrompt(
            promptText
        );

    if (inputTensor === null) {

        alert(
            "Mindestens drei bekannte Wörter eingeben."
        );

        return;
    }

    const predictionTensor =
        model.predict(
            inputTensor
        );

    const predictionValues =
        await predictionTensor.data();

    const predictions = [];

    for (let i = 0; i < predictionValues.length; i++) {

        predictions.push({

            word: indexToWord[i],

            probability:
                predictionValues[i]
        });
    }

    predictions.sort(

        function(a, b) {

            return (
                b.probability -
                a.probability
            );
        }
    );

    lastPredictions =
        predictions;

    showPredictions(
        predictions.slice(0, 10)
    );

    inputTensor.dispose();

    predictionTensor.dispose();
}


// Vorhersagen anzeigen

function showPredictions(predictions) {

    const output =
        document.getElementById(
            "prediction-output"
        );

    output.innerHTML = "";

    for (const prediction of predictions) {

        const button =
            document.createElement(
                "button"
            );

        button.className =
            "prediction-item";

        button.textContent =
            prediction.word +
            " (" +
            (
                prediction.probability
                * 100
            ).toFixed(2)
            + "%)";

        button.addEventListener(

            "click",

            async function () {

                addWordToPrompt(
                    prediction.word
                );

                await predictNextWords();
            }
        );

        output.appendChild(button);
    }
}


// Wort an Prompt anhängen

function addWordToPrompt(word) {

    const promptInput =
        document.getElementById(
            "prompt-input"
        );

    let text =
        promptInput.value.trim();

    if (text.length === 0) {

        promptInput.value =
            word;

    } else {

        promptInput.value =
            text + " " + word;
    }
}


// Bestes Wort übernehmen

async function acceptBestPrediction() {

    if (lastPredictions.length === 0) {

        await predictNextWords();
    }

    if (lastPredictions.length > 0) {

        const bestWord =
            lastPredictions[0].word;

        addWordToPrompt(
            bestWord
        );

        await predictNextWords();
    }
}


// Automatische Textgenerierung

async function startAutoGeneration() {

    autoRunning = true;

    for (let i = 0; i < 10; i++) {

        if (autoRunning === false) {

            break;
        }

        await acceptBestPrediction();

        await new Promise(function(resolve) {

            setTimeout(resolve, 500);
        });
    }

    autoRunning = false;
}


// Automatische Textgenerierung stoppen

function stopAutoGeneration() {

    autoRunning = false;
}


// Buttons

const trainButton =
    document.getElementById(
        "train-button"
    );

const predictButton =
    document.getElementById(
        "predict-button"
    );

const nextButton =
    document.getElementById(
        "next-button"
    );

const autoButton =
    document.getElementById(
        "auto-button"
    );

const stopButton =
    document.getElementById(
        "stop-button"
    );

const resetButton =
    document.getElementById(
        "reset-button"
    );


trainButton.addEventListener(

    "click",

    async function () {

        await trainModel();
    }
);


predictButton.addEventListener(

    "click",

    async function () {

        await predictNextWords();
    }
);


nextButton.addEventListener(

    "click",

    async function () {

        await acceptBestPrediction();
    }
);


autoButton.addEventListener(

    "click",

    async function () {

        await startAutoGeneration();
    }
);


stopButton.addEventListener(

    "click",

    function () {

        stopAutoGeneration();
    }
);


resetButton.addEventListener(

    "click",

    function () {

        console.log("Reset");
    }
);