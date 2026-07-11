const API_URL = "http://127.0.0.1:8000/evaluate";

const button = document.getElementById("evaluateBtn");

button.addEventListener("click", evaluateResponse);

async function evaluateResponse() {

    const question = document.getElementById("question").value.trim();
    const answer = document.getElementById("answer").value.trim();
    const reference = document.getElementById("reference").value.trim();

    if (!question || !answer || !reference) {
        alert("Please fill all fields.");
        return;
    }

    button.innerHTML = "⏳ Evaluating...";
    button.disabled = true;

    try {

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question,
                answer: answer,
                reference_answer: reference
            })

        });

        if (!response.ok) {
            throw new Error("Server Error");
        }

        const data = await response.json();

        console.log(data);

        //---------------------------------------
        // Evaluation Metrics
        //---------------------------------------

        const evaluation = data.evaluation;

        document.getElementById("accuracy").innerHTML =
            formatScore(evaluation.accuracy);

        document.getElementById("relevance").innerHTML =
            formatScore(evaluation.relevance);

        document.getElementById("faithfulness").innerHTML =
            formatScore(evaluation.faithfulness);

        document.getElementById("completeness").innerHTML =
            formatScore(evaluation.completeness);

        document.getElementById("hallucination").innerHTML =
            evaluation.hallucination_risk;

        document.getElementById("overall").innerHTML =
            formatScore(evaluation.overall_score);

        //---------------------------------------
        // Retrieved Context
        //---------------------------------------

        const contextDiv = document.getElementById("contexts");
        contextDiv.innerHTML = "";

        if (data.retrieved_context && data.retrieved_context.length > 0) {

            data.retrieved_context.forEach((chunk, index) => {

                const div = document.createElement("div");

                div.className = "context-item";

                div.innerHTML = `
                    <h4>Chunk ${index + 1}</h4>
                    <p>${chunk}</p>
                `;

                contextDiv.appendChild(div);

            });

        } else {

            contextDiv.innerHTML = `
                <div class="context-item">
                    <p>No context retrieved.</p>
                </div>
            `;

        }

    }

    catch (error) {

        console.error(error);

        alert("Error connecting to backend.");

    }

    finally {

        button.innerHTML = "⚡ Evaluate Response";
        button.disabled = false;

    }

}

//-------------------------------------
// Convert decimal score to percentage
//-------------------------------------

function formatScore(value) {

    if (value === undefined || value === null)
        return "--";

    if (typeof value === "string")
        return value;

    return (value * 100).toFixed(1) + "%";

}