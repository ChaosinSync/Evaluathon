# 🔒 Offline Multimodal RAG

### Privacy-First, Evidence-Grounded Document Intelligence

Offline Multimodal RAG is a locally hosted AI-powered document intelligence system that allows users to upload documents and ask questions about their content without sending sensitive data to cloud AI services.

The system supports **PDF, DOCX, and image-based documents**, retrieves relevant evidence locally, and uses a locally running **Qwen 2.5 1.5B** model through **Ollama** to generate grounded answers.

---

## 🚀 Key Features

- 📄 PDF document processing
- 📝 DOCX document processing
- 🖼️ PNG/JPG image support
- 🔤 OCR using Tesseract
- 🔎 Fast local evidence retrieval
- 🤖 Local Qwen 2.5 1.5B LLM
- 🔒 Privacy-preserving local processing
- 🌐 Works without an internet connection during runtime
- 📌 Page-level source references
- 🛡️ Evidence-grounded answers
- ❌ Prevents unsupported answers and hallucinations

---

## 💡 Problem

Sensitive documents often contain confidential technical, organizational, or personal information.

Traditional AI document assistants commonly depend on cloud-based APIs, requiring documents to leave the user's system.

At the same time, manually searching through lengthy documents is slow and inefficient.

This project addresses both problems by bringing document intelligence and AI inference to the user's local machine.

---

## 💡 Solution

The system creates a local pipeline:

**Upload → Extract/OCR → Retrieve Evidence → Verify Grounding → Local LLM → Grounded Answer**

Instead of allowing the language model to answer freely, the system first retrieves relevant evidence from the uploaded document.

If sufficient supporting evidence cannot be found, the system responds:

> "Not found in the provided evidence."

This helps reduce unsupported AI-generated answers.

---

## 🏗️ Architecture

```text
                    USER
                      │
                      ▼
            ┌───────────────────┐
            │  Document Upload  │
            │ PDF / DOCX / IMG  │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ Document Processing│
            └─────────┬─────────┘
                      │
             ┌────────┴────────┐
             ▼                 ▼
      Text Extraction       OCR Engine
      PDF / DOCX            Tesseract
             │                 │
             └────────┬────────┘
                      ▼
            ┌───────────────────┐
            │ Local Retrieval   │
            │ Keyword + Factual │
            │ Evidence Matching │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ Grounding Check   │
            └─────────┬─────────┘
                 ┌────┴────┐
                 │         │
              Evidence   No Evidence
                 │         │
                 ▼         ▼
          ┌────────────┐  "Not Found in
          │ Local Qwen │   Evidence"
          │ 2.5 1.5B  │
          │ + Ollama   │
          └──────┬─────┘
                 │
                 ▼
          ┌────────────────┐
          │ Grounded Answer│
          │ + Source Page  │
          └────────────────┘
