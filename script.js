const API_BASE = "http://127.0.0.1:8000";
const EVALUATE_URL = `${API_BASE}/evaluate`;
const HISTORY_URL = `${API_BASE}/history`;

const button = document.getElementById("evaluateBtn");
const pdfInput = document.getElementById("sourcePdf");
const clearPdfBtn = document.getElementById("clearPdfBtn");
const pdfFileName = document.getElementById("pdfFileName");

button.addEventListener("click", evaluateResponse);

//-------------------------------------
// Tab navigation
//-------------------------------------

const navLinks = document.querySelectorAll(".nav-link");
const tabPages = document.querySelectorAll(".tab-page");

navLinks.forEach((link) => {
    link.addEventListener("click", () => {
        const targetId = link.getAttribute("data-tab");

        navLinks.forEach((l) => l.classList.remove("active"));
        tabPages.forEach((p) => p.classList.remove("active"));

        link.classList.add("active");
        document.getElementById(targetId).classList.add("active");
    });
});

//-------------------------------------
// Show selected filename, or clear it
//-------------------------------------

pdfInput.addEventListener("change", () => {
    if (pdfInput.files.length > 0) {
        pdfFileName.textContent = `Selected: ${pdfInput.files[0].name}`;
    } else {
        pdfFileName.textContent = "";
    }
});

clearPdfBtn.addEventListener("click", () => {
    pdfInput.value = "";
    pdfFileName.textContent = "";
});

//-------------------------------------
// Run an evaluation
//-------------------------------------

async function evaluateResponse() {

    const question = document.getElementById("question").value.trim();
    const answer = document.getElementById("answer").value.trim();
    const reference = document.getElementById("reference").value.trim();
    const pdfFile = pdfInput.files.length > 0 ? pdfInput.files[0] : null;

    if (!question || !answer) {
        alert("Please fill in the question and candidate answer.");
        return;
    }

    button.innerHTML = "⏳ Evaluating...";
    button.disabled = true;

    try {

        const formData = new FormData();
        formData.append("question", question);
        formData.append("answer", answer);

        if (reference) {
            formData.append("reference_answer", reference);
        }

        if (pdfFile) {
            formData.append("source_pdf", pdfFile);
        }

        const response = await fetch(EVALUATE_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Server Error");
        }

        const data = await response.json();

        console.log(data);

        const evaluation = data.evaluation;

        document.getElementById("accuracy").innerHTML =
            formatScore(evaluation.accuracy);

        document.getElementById("accuracyEvidence").innerHTML =
            evaluation.accuracy_evidence || "";

        document.getElementById("accuracyExcerpt").innerHTML =
            evaluation.accuracy_supporting_excerpt
                ? `“${evaluation.accuracy_supporting_excerpt}”`
                : "";

        document.getElementById("relevance").innerHTML =
            formatScore(evaluation.relevance);

        document.getElementById("relevanceReasoning").innerHTML =
            evaluation.relevance_reasoning || "";

        document.getElementById("hallucination").innerHTML =
            evaluation.hallucination_risk;

        document.getElementById("hallucinationReasoning").innerHTML =
            evaluation.hallucination_reasoning || "";

        renderFlaggedStatements(evaluation.hallucination_flagged_statements);

        document.getElementById("overall").innerHTML =
            formatScore(evaluation.overall_score);

        renderContexts(data.retrieved_context, data.used_source_pdf);

        loadHistory();

    }

    catch (error) {

        console.error(error);

        alert("Error connecting to backend.");

    }

    finally {

        button.innerHTML = " Evaluate Response";
        button.disabled = false;

    }

}

//-------------------------------------
// Render hallucination flagged statements list
//-------------------------------------

function renderFlaggedStatements(statements) {

    const list = document.getElementById("hallucinationFlags");
    list.innerHTML = "";

    if (!statements || statements.length === 0) {
        return;
    }

    statements.forEach((statement) => {
        const li = document.createElement("li");
        li.textContent = statement;
        list.appendChild(li);
    });

}

//-------------------------------------
// Render retrieved context chunks (on the Retrieval tab)
//-------------------------------------

function renderContexts(chunks, usedSourcePdf) {

    const contextDiv = document.getElementById("contexts");
    contextDiv.innerHTML = "";

    if (chunks && chunks.length > 0) {

        chunks.forEach((chunk, index) => {

            const div = document.createElement("div");
            div.className = "context-item";

            const isPdfChunk = usedSourcePdf && index === chunks.length - 1;
            const label = isPdfChunk ? "Uploaded Source PDF" : `Chunk ${index + 1}`;

            div.innerHTML = `
                <h4>${label}</h4>
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

//-------------------------------------
// Load and render evaluation history (on the History tab)
//-------------------------------------

async function loadHistory() {

    const historyList = document.getElementById("historyList");

    try {

        const response = await fetch(`${HISTORY_URL}?limit=20`);

        if (!response.ok) {
            throw new Error("Failed to load history");
        }

        const data = await response.json();
        const records = data.history || [];

        if (records.length === 0) {
            historyList.innerHTML = `
                <div class="context-item">
                    <p>No evaluations yet.</p>
                </div>
            `;
            return;
        }

        historyList.innerHTML = "";

        records.forEach((record) => {

            const div = document.createElement("div");
            div.className = "context-item";

            const refTag = record.reference_answer ? "" : " (no reference)";
            const pdfTag = record.used_source_pdf ? " · PDF used" : "";

            const flags = record.hallucination_flagged_statements || [];
            const flagsHtml = flags.length > 0
                ? `<ul class="flag-list">${flags.map(f => `<li>${f}</li>`).join("")}</ul>`
                : "";

            div.innerHTML = `
                <h4>${formatDate(record.created_at)}${refTag}${pdfTag}</h4>
                <p><strong>Q:</strong> ${record.question}</p>
                <p><strong>A:</strong> ${record.answer}</p>
                <p>
                    Accuracy: ${formatScore(record.accuracy)} ·
                    Relevance: ${formatScore(record.relevance)} ·
                    Hallucination: ${record.hallucination_risk} ·
                    Overall: ${formatScore(record.overall_score)}
                </p>
                <p class="metric-reasoning">${record.relevance_reasoning || ""}</p>
                <p class="metric-reasoning">${record.accuracy_evidence || ""}</p>
                <p class="metric-reasoning">${record.hallucination_reasoning || ""}</p>
                ${flagsHtml}
            `;

            historyList.appendChild(div);

        });

    } catch (error) {

        console.error(error);
        historyList.innerHTML = `
            <div class="context-item">
                <p>Could not load history.</p>
            </div>
        `;

    }

}

document.addEventListener("DOMContentLoaded", loadHistory);

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

//-------------------------------------
// Format an ISO timestamp for display
//-------------------------------------

function formatDate(isoString) {

    if (!isoString) return "Unknown time";

    const date = new Date(isoString);

    return date.toLocaleString();

}