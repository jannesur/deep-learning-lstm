console.log("Projekt gestartet");

console.log("TensorFlow.js geladen:", tf);
console.log("Plotly geladen:", Plotly);

let dictionary = [];

let wordToIndex = {};

let indexToWord = {};

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

    for (
        let i = 0;
        i < words.length - sequenceLength;
        i++
    ) {

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

        inputs: inputs,

        labels: labels
    };
}


// Trainingstext vorbereiten

function prepareTrainingText() {

    const text =
        document
            .getElementById("prompt-input")
            .value;

    const words =
        splitIntoWords(text);

    createDictionary(words);

    const trainingData =
        createTrainingData(words);

    console.log(
        "Anzahl Wörter:",
        words.length
    );

    console.log(
        "Dictionary:",
        dictionary
    );

    console.log(
        "wordToIndex:",
        wordToIndex
    );

    console.log(
        "indexToWord:",
        indexToWord
    );

    console.log(
        "Trainingsdaten:",
        trainingData
    );
}


// Buttons

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


// Vorhersage

predictButton.addEventListener(

    "click",

    function () {

        prepareTrainingText();

        console.log(
            "Trainingssequenzen erstellt"
        );
    }
);


// Platzhalter

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