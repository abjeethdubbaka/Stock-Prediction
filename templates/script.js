document.getElementById("predictionForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const ticker = formData.get("ticker");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();
        document.getElementById("result").innerHTML = `
            <p>${result.message}</p>
            <p>Check terminal/logs for detailed predictions.</p>
        `;
    } catch (error) {
        document.getElementById("result").innerHTML = `
            <p style="color: red;">Error: ${error.message}</p>
        `;
    }
});