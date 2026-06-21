require('dotenv').config();
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const Groq = require('groq-sdk');
const RAGEngine = require('./lib/rag');
const DatasetManager = require('./lib/dataset');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
});

const ragEngine = new RAGEngine();
const datasetManager = new DatasetManager();

// Memori penyimpan riwayat obrolan berbasis sesi website agar percakapan nyambung
const chatHistories = new Map();

const knowledgeFile = path.join(__dirname, 'knowledge.json');
const behaviorFile = path.join(__dirname, 'config', 'behavior.json');

if (!fs.existsSync(knowledgeFile)) {
    fs.writeFileSync(knowledgeFile, JSON.stringify({ keywords: {}, responses: {} }, null, 2));
}

function loadKnowledge() {
    try {
        const data = fs.readFileSync(knowledgeFile, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        console.error('Error loading knowledge:', error);
        return { keywords: {}, responses: {} };
    }
}

function saveKnowledge(data) {
    try {
        fs.writeFileSync(knowledgeFile, JSON.stringify(data, null, 2));
        ragEngine.clearCache();
        return true;
    } catch (error) {
        console.error('Error saving knowledge:', error);
        return false;
    }
}

function loadBehavior() {
    try {
        if (!fs.existsSync(behaviorFile)) return null;
        const content = fs.readFileSync(behaviorFile, 'utf8');
        return JSON.parse(content);
    } catch (error) {
        console.error('Error loading behavior config:', error.message);
        return null;
    }
}

function saveBehavior(obj) {
    try {
        fs.mkdirSync(path.dirname(behaviorFile), { recursive: true });
        fs.writeFileSync(behaviorFile, JSON.stringify(obj, null, 2));
        return true;
    } catch (error) {
        console.error('Error saving behavior config:', error.message);
        return false;
    }
}

/**
 * Memproses pesan ke API Groq menggunakan Konteks RAG dan Riwayat Obrolan
 */
async function getAIResponse(message, contextItems = [], behavior = null, userId) {
    try {
        const contextBlock = ragEngine.buildContextBlock(contextItems);
        if (!behavior) {
            behavior = loadBehavior() || {
                system_instructions: 'Jawab berdasarkan konteks.',
                fallback_response: 'Mohon maaf, data tidak ditemukan.',
                max_sentences: 2,
                language: 'id'
            };
        }

        const contextText = contextItems.length > 0 
            ? `\n\nKonteks Dokumen Akademik TUS Resmi:\n${contextBlock}` 
            : `\n\nKonteks Dokumen Akademik TUS: [Tidak ada aturan akademik spesifik yang relevan dengan pertanyaan mahasiswa saat ini]`;

        const systemParts = [];
        if (behavior.system_instructions) systemParts.push(behavior.system_instructions);
        
        systemParts.push(`\nPanduan Ekstra: Jika pertanyaan menanyakan produk atau hal yang tidak ada di Konteks Data Akademik, katakan: "${behavior.fallback_response}"`);
        systemParts.push(`Jawab maksimal ${behavior.max_sentences || 3} kalimat. Bahasa: ${behavior.language || 'id'}.`);
        
        const systemMessage = systemParts.join(' ') + contextText;

        // Ambil riwayat obrolan sebelumnya untuk pengguna web ini
        let history = chatHistories.get(userId) || [];

        // Siapkan array messages untuk Groq
        const messages = [
            { role: 'system', content: systemMessage },
            ...history,
            { role: 'user', content: message }
        ];

        const completion = await groq.chat.completions.create({
            messages: messages,
            model: process.env.GROQ_MODEL || 'llama-3.1-8b-instant',
            max_tokens: Number(process.env.GROQ_MAX_TOKENS || 200),
            temperature: 0.2
        });

        const aiResponseText = completion.choices[0].message.content;

        // Simpan pesan saat ini ke riwayat obrolan lokal memori
        history.push({ role: 'user', content: message });
        history.push({ role: 'assistant', content: aiResponseText });

        // Batasi memori hanya 10 pesan terakhir (5 tanya, 5 jawab) agar tidak bengkak
        if (history.length > 10) {
            history = history.slice(history.length - 10);
        }
        
        chatHistories.set(userId, history);

        return aiResponseText;
    } catch (error) {
        console.error('Error getting AI response:', error.message);
        return null;
    }
}

// ====================================================================
// ENDPOINT UTAMA: Menangani request chat langsung dari Website Frontend
// ====================================================================
app.post('/api/chat', async (req, res) => {
    try {
        const { message, userId } = req.body;
        const activeUserId = userId || 'default-web-user';

        if (!message || message.trim() === "") {
            return res.status(400).json({ error: 'Pesan tidak boleh kosong', success: false });
        }

        console.log(`[Web Message] User (${activeUserId}): ${message}`);

        const knowledge = loadKnowledge();
        const pesanMasuk = message.toLowerCase().trim();
        let faqResponse = null;

        // 1. GARDA TERDEPAN: Cek Kata Kunci Kaku (knowledge.json)
        if (knowledge.keywords && knowledge.responses) {
            for (const [kunciUtama, daftarKata] of Object.entries(knowledge.keywords)) {
                if (daftarKata.some(kata => pesanMasuk.includes(kata))) {
                    faqResponse = knowledge.responses[kunciUtama];
                    break;
                }
            }
        }

        if (faqResponse) {
            console.log(`[Web Chat] Match via knowledge.json`);
            return res.json({ reply: faqResponse, source: 'FAQ Direct Match' });
        }

        // 2. DETEKSI SAPAAN RINGAN (Mencegah Greeting Loop)
        // Jika hanya menyapa "halo", "hai", atau "p", langsung jawab tanpa perlu mencari RAG/Groq bengkak
        const sapaanSederhana = ['halo', 'hai', 'p', 'permisi', 'selamat pagi', 'selamat siang', 'assalamualaikum'];
        if (sapaanSederhana.includes(pesanMasuk)) {
            return res.json({ 
                reply: 'Halo! Ada yang bisa saya bantu terkait informasi administrasi akademik kampus Telkom University Surabaya?', 
                source: 'Static Greeting' 
            });
        }

        // 3. JALUR UTAMA: Ekstraksi Dokumen Menggunakan RAG TF-IDF Lokal
        const allDocuments = datasetManager.getAllDocuments();
        
        // Ambil riwayat chat terakhir untuk digabungkan ke query agar memahami kata "itu", "tersebut", "ini"
        let history = chatHistories.get(activeUserId) || [];
        let contextualQuery = message;
        
        // Jika ada percakapan sebelumnya, gabungkan topik terakhir agar TF-IDF Lokal tahu konteksnya
        if (history.length > 0) {
            const lastUserMsg = history[history.length - 2]?.content || "";
            contextualQuery = `${lastUserMsg} ${message}`;
        }

        const contextItems = ragEngine.retrieveContext(
            contextualQuery,
            allDocuments,
            Number(process.env.RAG_TOP_K || 3)
        );
        
        console.log(`[Web Chat] RAG Retrieved ${contextItems.length} context(s) untuk kueri: "${contextualQuery}"`);
        
        // 4. KIRIM KE GROQ LLM (Otak utama yang merangkai kalimat)
        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('AI response timeout')), 20000)
        );
        
        try {
            const behavior = loadBehavior();
            
            const aiResponse = await Promise.race([
                getAIResponse(message, contextItems, behavior, activeUserId),
                timeoutPromise
            ]);

            if (aiResponse && !aiResponse.toLowerCase().includes("halo! saya adalah smart chatbot")) {
                return res.json({ reply: aiResponse, source: 'RAG Engine + Groq AI' });
            } else {
                // Jika AI error atau mengembalikan pesan sapaan kosong, berikan fallback solutif
                return res.json({ 
                    reply: 'Mohon maaf, saya belum menemukan aturan spesifik mengenai hal tersebut di buku pedoman akademik saat ini. Bisa tolong perjelas pertanyaannya?', 
                    source: 'Safe Fallback' 
                });
            }
        } catch (aiError) {
            console.error('AI Processing Error:', aiError.message);
            return res.json({ 
                reply: 'Maaf, terjadi antrean komputasi di server AI. Silakan coba kirimkan ulang pertanyaan Anda dalam beberapa saat.', 
                source: 'Timeout Fallback' 
            });
        }
    } catch (error) {
        console.error('API Chat Error:', error.message);
        return res.status(500).json({ error: 'Internal Server Error' });
    }
});

// ====================================================================
// ENDPOINT MANAJEMEN DATASET & CONFIG (UNTUK DASHBOARD ADMIN)
// ====================================================================

app.get('/api/datasets', (req, res) => {
    res.json({
        datasets: datasetManager.listDatasets(),
        totalDocuments: datasetManager.getAllDocuments().length
    });
});

app.get('/api/datasets/:name', (req, res) => {
    const docs = datasetManager.getDatasetDocuments(req.params.name);
    if (docs.length === 0) {
        return res.status(404).json({ message: 'Dataset tidak ditemukan' });
    }
    res.json({ documents: docs });
});

app.post('/api/datasets', (req, res) => {
    try {
        const { name, data } = req.body;
        if (!name || !data) {
            return res.status(400).json({ message: 'name dan data harus diisi' });
        }
        const result = datasetManager.saveDataset(name, data);
        res.json(result);
    } catch (error) {
        res.status(500).json({ message: 'Error: ' + error.message });
    }
});

app.get('/api/knowledge/keywords', (req, res) => {
    const knowledge = loadKnowledge();
    res.json(knowledge);
});

app.post('/api/knowledge/keyword', (req, res) => {
    try {
        const { keyword, response } = req.body;
        if (!keyword || !response) {
            return res.status(400).json({ message: 'Keyword dan response harus diisi', success: false });
        }
        const knowledge = loadKnowledge();
        knowledge.responses[keyword.toLowerCase().trim()] = response;
        if (saveKnowledge(knowledge)) {
            res.json({ message: 'Keyword berhasil disimpan', success: true });
        } else {
            res.status(500).json({ message: 'Error menyimpan keyword', success: false });
        }
    } catch (error) {
        res.status(500).json({ message: 'Error: ' + error.message, success: false });
    }
});

app.delete('/api/knowledge/keyword/:keyword', (req, res) => {
    try {
        const keyword = decodeURIComponent(req.params.keyword).toLowerCase();
        const knowledge = loadKnowledge();
        if (knowledge.responses[keyword]) {
            delete knowledge.responses[keyword];
            if (saveKnowledge(knowledge)) {
                res.json({ message: 'Keyword berhasil dihapus', success: true });
            } else {
                res.status(500).json({ message: 'Error menghapus keyword', success: false });
            }
        } else {
            res.status(404).json({ message: 'Keyword tidak ditemukan', success: false });
        }
    } catch (error) {
        res.status(500).json({ message: 'Error: ' + error.message, success: false });
    }
});

app.get('/api/behavior', (req, res) => {
    try {
        const behavior = loadBehavior();
        if (!behavior) return res.status(404).json({ message: 'Behavior config not found' });
        res.json(behavior);
    } catch (error) {
        res.status(500).json({ message: 'Error: ' + error.message });
    }
});

app.post('/api/behavior', (req, res) => {
    try {
        const obj = req.body;
        if (!obj || typeof obj !== 'object') {
            return res.status(400).json({ message: 'Invalid behavior object' });
        }
        const saved = saveBehavior(obj);
        if (saved) return res.json({ message: 'Behavior saved', success: true });
        res.status(500).json({ message: 'Error saving behavior', success: false });
    } catch (error) {
        res.status(500).json({ message: 'Error: ' + error.message });
    }
});

app.listen(PORT, () => {
    console.log(`=========================================`);
    console.log(`Server BERJALAN murni berbasis Website.`);
    console.log(`Akses Chat App: http://localhost:${PORT}`);
    console.log(`Akses Panel Admin: http://localhost:${PORT}/admin.html`);
    console.log(`=========================================`);
    
    // Membangun indeks peta TF-IDF lokal saat server pertama kali menyala (Instan < 1 detik)
    const activeDocs = datasetManager.getAllDocuments();
    ragEngine.buildVectorIndex(activeDocs);
    console.log(`Datasets loaded: ${datasetManager.listDatasets().length} file CSV.`);
});