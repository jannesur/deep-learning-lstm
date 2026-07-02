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

    if (text.trim().length === 0) {

        alert(
            "Bitte Trainingsdaten eingeben."
        );

        return;
    }

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

    if (model !== undefined) {

        model.dispose();
    }

    model = createModel();

    lossHistory = [];

    lastPredictions = [];

    document.getElementById(
        "prediction-output"
    ).innerHTML = "";

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

    calculateTopKAccuracy(words);

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

    if (promptText.trim().length === 0) {

        alert(
            "Bitte einen Text eingeben."
        );

        return;
    }

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


// Gewichtete Auswahl aus den Top-5

async function acceptWeightedTopPrediction() {

    if (lastPredictions.length === 0) {

        await predictNextWords();
    }

    if (lastPredictions.length === 0) {

        return;
    }

    const topPredictions =
        lastPredictions.slice(0, 5);

    let probabilitySum = 0;

    for (const prediction of topPredictions) {

        probabilitySum =
            probabilitySum +
            prediction.probability;
    }

    let randomValue =
        Math.random() * probabilitySum;

    let selectedWord =
        topPredictions[0].word;

    for (const prediction of topPredictions) {

        randomValue =
            randomValue -
            prediction.probability;

        if (randomValue <= 0) {

            selectedWord =
                prediction.word;

            break;
        }
    }

    addWordToPrompt(
        selectedWord
    );

    await predictNextWords();
}


// Automatische Textgenerierung

async function startAutoGeneration() {

    if (autoRunning === true) {

        return;
    }

    autoRunning = true;

    for (let i = 0; i < 10; i++) {

        if (autoRunning === false) {

            break;
        }

        await acceptWeightedTopPrediction();

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


// Anwendung zurücksetzen

function resetApplication() {

    autoRunning = false;

    lastPredictions = [];

    dictionary = [];

    wordToIndex = {};

    indexToWord = {};

    lossHistory = [];

    if (model !== undefined) {

        model.dispose();

        model = undefined;
    }

    document.getElementById(
        "prompt-input"
    ).value = "";

    document.getElementById(
        "prediction-output"
    ).innerHTML = "";

    document.getElementById(
        "training-status"
    ).textContent =
        "Modell noch nicht trainiert.";

    document.getElementById(
        "accuracy-1"
    ).textContent = "-";

    document.getElementById(
        "accuracy-5"
    ).textContent = "-";

    document.getElementById(
        "accuracy-10"
    ).textContent = "-";

    document.getElementById(
        "accuracy-20"
    ).textContent = "-";

    document.getElementById(
        "accuracy-100"
    ).textContent = "-";

    Plotly.purge(
        "loss-plot"
    );
}


// Top-K Accuracy berechnen

function calculateTopKAccuracy(words) {

    let correctTop1 = 0;

    let correctTop5 = 0;

    let correctTop10 = 0;

    let correctTop20 = 0;

    let correctTop100 = 0;

    let total = 0;

    for (let i = 0; i < words.length - sequenceLength; i++) {

        const inputWords =
            words.slice(
                i,
                i + sequenceLength
            );

        const correctWord =
            words[i + sequenceLength];

        const prompt =
            inputWords.join(" ");

        const inputTensor =
            createInputFromPrompt(
                prompt
            );

        if (inputTensor === null) {

            continue;
        }

        const predictionTensor =
            model.predict(
                inputTensor
            );

        const predictionValues =
            predictionTensor.dataSync();

        const predictions = [];

        for (let j = 0; j < predictionValues.length; j++) {

            predictions.push({

                word: indexToWord[j],

                probability:
                    predictionValues[j]
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

        const top1 =
            predictions.slice(0, Math.min(1, predictions.length));

        const top5 =
            predictions.slice(0, Math.min(5, predictions.length));

        const top10 =
            predictions.slice(0, Math.min(10, predictions.length));

        const top20 =
            predictions.slice(0, Math.min(20, predictions.length));

        const top100 =
            predictions.slice(0, Math.min(100, predictions.length));

        if (top1.map(item => item.word).includes(correctWord)) {

            correctTop1++;
        }

        if (top5.map(item => item.word).includes(correctWord)) {

            correctTop5++;
        }

        if (top10.map(item => item.word).includes(correctWord)) {

            correctTop10++;
        }

        if (top20.map(item => item.word).includes(correctWord)) {

            correctTop20++;
        }

        if (top100.map(item => item.word).includes(correctWord)) {

            correctTop100++;
        }

        total++;

        inputTensor.dispose();

        predictionTensor.dispose();
    }

    if (total === 0) {

        document.getElementById("accuracy-1").textContent = "-";
        document.getElementById("accuracy-5").textContent = "-";
        document.getElementById("accuracy-10").textContent = "-";
        document.getElementById("accuracy-20").textContent = "-";
        document.getElementById("accuracy-100").textContent = "-";

        return;
    }

    document.getElementById("accuracy-1").textContent =
        (correctTop1 / total * 100).toFixed(2) + "%";

    document.getElementById("accuracy-5").textContent =
        (correctTop5 / total * 100).toFixed(2) + "%";

    document.getElementById("accuracy-10").textContent =
        (correctTop10 / total * 100).toFixed(2) + "%";

    document.getElementById("accuracy-20").textContent =
        (correctTop20 / total * 100).toFixed(2) + "%";

    document.getElementById("accuracy-100").textContent =
        (correctTop100 / total * 100).toFixed(2) + "%";
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

        resetApplication();
    }
);