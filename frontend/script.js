async function translateText() {
    const text = document.getElementById("inputText").value;
    const language = document.getElementById("languageSelect").value;
    const output = document.getElementById("outputText");
    const loading = document.getElementById("loading");

    if (!text) {
        alert("Please enter text");
        return;
    }

    loading.innerText = "Translating...";
    output.innerText = "";

    try {
        const response = await fetch("/translate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                language: language
            })
        });

        const data = await response.json();

        output.innerText = data.translation;

    } catch (error) {
        output.innerText = "Server error.";
    }

    loading.innerText = "";
}