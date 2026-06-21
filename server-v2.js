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

function getDefaultBehavior() {
    return {
        system_instructions: 'Jawab berdasarkan konteks.',
        fallback_response: 'Mohon maaf, informasi spesifik mengenai hal tersebut belum tercatat di sistem pedoman akademik kami. Silakan hubungi bagian Sekretariat Kampus atau Layanan Mahasiswa untuk info lebih lanjut.',
        max_sentences: 4,
        language: 'id'
    };
}

// =========================================================================
// KAMUS KOREKSI TYPO & BAHASA GAUL/SANTAI (Mencegah RAG 0 Context)
// Dipindah ke level module supaya tidak dibuat ulang setiap request,
// dan supaya bisa dipakai bersama oleh beberapa fungsi.
// =========================================================================
const kamusKoreksiMassal = {
    "yidisium": "yudisium",
    "yudisum": "yudisium",
    "yudis": "yudisium",
    "eprt": "eprt toefl",
    "epert": "eprt toefl",
    "tofel": "eprt toefl",
    "bpp": "bpp ukt uang kuliah",
    "ukt": "bpp ukt uang kuliah",
    "sksan": "sks maksimal kuota",
    "krsan": "krs ksm registrasi",
    "ksman": "krs ksm registrasi",
    "doswal": "dosen wali perwalian",
    "dosen wali": "dosen wali perwalian",
    "skripsian": "skripsi tugas akhir ta",
    "kp": "kerja praktik magang wrap",
    "internsip": "kerja praktik magang wrap",
    "cumlaud": "cum laude pujian",
    "comlaude": "cum laude pujian",
    "dropaut": "drop out sp surat peringatan",
    "mangkir": "mangkir tidak aktif nonaktif",
    "semester pendek": "semester antara pendek sp",
    "lks": "laporan kemajuan studi lks",
    "rapor": "laporan kemajuan studi lks",
    "khs": "kartu hasil studi khs",
    "transkrip": "transkrip akademik nilai",
    "3.5 tahun": "7 semester lulus cepat masa studi normal",
    "3,5 tahun": "7 semester lulus cepat masa studi normal",
    "3 setengah tahun": "7 semester lulus cepat masa studi normal",
    "4 tahun": "8 semester masa studi normal sarjana",
    "3 tahun": "6 semester masa studi normal diploma tiga",
    "7 semester": "7 semester lulus cepat masa studi normal",
    "8 semester": "8 semester masa studi normal sarjana",
    "6 semester": "6 semester masa studi normal diploma tiga",
    "nilai minimal": "nilai huruf terendah lulus minimum",
    "nilai kelulusan": "nilai huruf terendah lulus minimum",
    "ngulang matkul": "nilai d atau e mengulang mata kuliah",
    "perbaikan nilai": "nilai d atau e mengulang mata kuliah"
};

// Catatan: "ta" dan "sp" SENGAJA TIDAK dimasukkan ke kamus di atas sebagai key
// tunggal karena terlalu pendek & ambigu (bisa muncul sebagai pecahan kata lain
// saat replace), beda dengan kataBasaBasi yang dicek per-kata utuh.

const kataBasaBasi = [
    "bagaimana", "apakah", "gimana", "sih", "dong", "kak", "min", "tolong",
    "mau", "tanya", "saya", "kamu", "itu", "ini", "yang", "di", "ke", "dari",
    "bisa", "kah", "bila", "jika", "kalau", "tentang", "mengenai", "untuk",
    "buat", "ikut", "ada", "nanya", "ya", "kok", "nih", "syarat",
    "cara", "aturan", "ketentuan", "panduan", "adalah", "apa", "biar", "supaya"
];

const kamusEkspansiMaksimal = {
    "eprt": "eprt toefl ielts kecakapan bahasa inggris skor nilai minimum kelulusan lulus",
    "toefl": "eprt toefl ielts kecakapan bahasa inggris skor nilai minimum kelulusan lulus",
    "tak": "tak transkrip aktivitas kemahasiswaan poin minimal organisasi sertifikat",
    "yudisium": "yudisium dekan sidang penetapan kelulusan ijazah skl fakultas rektor",
    "wisuda": "syarat lulus kelulusan wisuda ukt lunas publikasi ta ijazah transkrip",
    "skripsi": "tugas akhir ta skripsi skripsian sidang proposal pembimbing artikel ilmiah",
    "kp": "magang kerja praktik kp wrap internship kerja industri",
    "magang": "magang kerja praktik kp wrap internship kerja industri",
    "cuti": "cuti akademik nonaktif bpp status 10 persen tingkat 1 izin pimpinan upps",
    "sks": "beban belajar sks maksimal kuota ip ips ambil krs semester",
    "krs": "krs ksm registrasi daftar ulang ukt ksm cetak awal semester",
    "do": "drop out sp surat peringatan evaluasi tingkat do spa sanksi akademik",
    "nilai": "skala nilai bobot indeks mutu konversi a ab b bc c d e terendah",
    "lks": "laporan kemajuan studi lks orang tua broadcast nilai evaluasi",
    "fast track": "fast track skema studi percepatan sarjana magister 10 semester ipk 3.25"
};

/**
 * Memproses satu pesan mentah jadi query RAG yang kaya (typo correction,
 * stopword removal, ekspansi sinonim, dan penyelarasan riwayat percakapan).
 * Dipisah jadi fungsi sendiri supaya tidak ada duplikasi variabel/logic
 * seperti pada versi sebelumnya, dan supaya gampang diuji terpisah.
 */
function buildEnrichedQuery(rawMessage, activeUserId) {
    let pesanMasuk = rawMessage.toLowerCase().trim();

    // LAYER 1: Koreksi typo & bahasa gaul
    for (const [salah, benar] of Object.entries(kamusKoreksiMassal)) {
        if (pesanMasuk.includes(salah)) {
            const pattern = salah.includes(' ')
                ? salah.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
                : `\\b${salah}\\b`;
            pesanMasuk = pesanMasuk.replace(new RegExp(pattern, 'g'), benar);
        }
    }

    // LAYER 2: Pembersihan stopword
    const kataKunciInti = pesanMasuk
        .split(/\s+/)
        .filter(kata => !kataBasaBasi.includes(kata))
        .join(' ');

    // LAYER 3: Ekspansi sinonim akademik
    let queryDibersihkan = kataKunciInti;
    for (const [singkatan, deskripsiPanjang] of Object.entries(kamusEkspansiMaksimal)) {
        if (pesanMasuk.includes(singkatan) || queryDibersihkan.includes(singkatan)) {
            queryDibersihkan = `${queryDibersihkan} ${deskripsiPanjang}`;
        }
    }

    // LAYER 4: Penyelarasan dengan riwayat chat (biar nyambung kalau user nanya lanjutan)
    const history = chatHistories.get(activeUserId) || [];
    if (history.length > 0) {
        const lastUserMsg = history[history.length - 2]?.content || '';
        const cleanedLastMsg = lastUserMsg
            .toLowerCase()
            .split(/\s+/)
            .filter(k => !kataBasaBasi.includes(k))
            .join(' ');
        queryDibersihkan = `${cleanedLastMsg} ${queryDibersihkan}`;
    }

    return queryDibersihkan.replace(/\s+/g, ' ').trim();
}

/**
 * Mencoba mencocokkan pesan ke knowledge.json (FAQ kaku: sapaan, terima kasih, dll).
 * Word-boundary match untuk kata tunggal, substring match untuk frasa multi-kata.
 */
function matchKnowledgeBase(pesanMasuk, knowledge) {
    if (!knowledge.keywords || !knowledge.responses) return null;

    const potonganKataUser = pesanMasuk.split(/\s+/);

    for (const [kunciUtama, daftarKata] of Object.entries(knowledge.keywords)) {
        const adaMencocok = daftarKata.some(kataDariJson => {
            if (kataDariJson.includes(' ')) {
                return pesanMasuk.includes(kataDariJson);
            }
            return potonganKataUser.includes(kataDariJson);
        });

        if (adaMencocok) {
            return knowledge.responses[kunciUtama];
        }
    }

    return null;
}

/**
 * Memproses pesan ke API Groq menggunakan Konteks RAG dan Riwayat Obrolan.
 * PENTING: fallback_response TIDAK diserahkan ke LLM untuk "diucapkan ulang" --
 * itu rawan diparafrase/diabaikan oleh model kecil. Kalau context kosong,
 * server-v2.js akan langsung balas fallback_response tanpa memanggil fungsi ini
 * sama sekali (lihat endpoint /api/chat). Fungsi ini hanya dipanggil ketika
 * context RAG memang ada isinya.
 */
async function getAIResponse(message, contextItems, behavior, userId) {
    try {
        const contextBlock = ragEngine.buildContextBlock(contextItems);

        const systemParts = [];
        if (behavior.system_instructions) systemParts.push(behavior.system_instructions);
        systemParts.push(`Jawab HANYA berdasarkan Konteks Dokumen Akademik di bawah ini. Jangan mengarang informasi yang tidak ada di konteks.`);
        systemParts.push(`Jawab maksimal ${behavior.max_sentences || 4} kalimat. Bahasa: ${behavior.language || 'id'}.`);

        const systemMessage = `${systemParts.join(' ')}\n\nKonteks Dokumen Akademik TUS Resmi:\n${contextBlock}`;

        let history = chatHistories.get(userId) || [];

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

        history.push({ role: 'user', content: message });
        history.push({ role: 'assistant', content: aiResponseText });

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

/**
 * Menyimpan giliran fallback ke riwayat juga, supaya chatHistories tetap
 * konsisten merepresentasikan percakapan (termasuk saat bot fallback),
 * dan supaya Layer 4 (penyelarasan riwayat) tidak nyasar ke giliran lama.
 */
function recordFallbackTurn(userId, userMessage, fallbackText) {
    let history = chatHistories.get(userId) || [];
    history.push({ role: 'user', content: userMessage });
    history.push({ role: 'assistant', content: fallbackText });
    if (history.length > 10) {
        history = history.slice(history.length - 10);
    }
    chatHistories.set(userId, history);
}

// ====================================================================
// ENDPOINT UTAMA: Menangani request chat langsung dari Website Frontend
// ====================================================================
app.post('/api/chat', async (req, res) => {
    try {
        const { message, userId } = req.body;
        const activeUserId = userId || 'default-web-user';

        if (!message || message.trim() === '') {
            return res.status(400).json({ error: 'Pesan tidak boleh kosong', success: false });
        }

        console.log(`[Web Message] User (${activeUserId}): ${message}`);

        const knowledge = loadKnowledge();
        const pesanMasuk = message.toLowerCase().trim();

        // =====================================================================
        // 1. GARDA TERDEPAN: Cek Kata Kunci Kaku (FAQ: sapaan, terima kasih, dll)
        // =====================================================================
        const faqResponse = matchKnowledgeBase(pesanMasuk, knowledge);
        if (faqResponse) {
            console.log('[Web Chat] Match via knowledge.json');
            recordFallbackTurn(activeUserId, message, faqResponse);
            return res.json({ reply: faqResponse, source: 'FAQ Direct Match' });
        }

        // =====================================================================
        // 2. JALUR UTAMA: Ekstraksi Dokumen Menggunakan RAG TF-IDF Lokal
        // =====================================================================
        const allDocuments = datasetManager.getAllDocuments();
        const queryDibersihkan = buildEnrichedQuery(message, activeUserId);

        const contextItems = ragEngine.retrieveContext(
            queryDibersihkan,
            allDocuments,
            Number(process.env.RAG_TOP_K || 6)
        );

        console.log(`[Web Chat] RAG Final Extracted Query: "${queryDibersihkan}" -> Retrieved ${contextItems.length} context(s)`);

        const behavior = loadBehavior() || getDefaultBehavior();

        // =====================================================================
        // 3. KONTEKS KOSONG -> langsung balas fallback yang KONSISTEN,
        //    tanpa memanggil Groq sama sekali. Ini mencegah LLM mengarang
        //    kalimat fallback sendiri (yang sebelumnya menyebabkan respons
        //    tidak konsisten seperti di screenshot).
        // =====================================================================
        if (contextItems.length === 0) {
            console.log('[Web Chat] Context kosong -> fallback langsung (tanpa panggil Groq)');
            recordFallbackTurn(activeUserId, message, behavior.fallback_response);
            return res.json({ reply: behavior.fallback_response, source: 'No Context Fallback' });
        }

        // =====================================================================
        // 4. CONTEXT DITEMUKAN -> tanya Groq dengan timeout guard
        // =====================================================================
        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('AI response timeout')), 20000)
        );

        try {
            const aiResponse = await Promise.race([
                getAIResponse(message, contextItems, behavior, activeUserId),
                timeoutPromise
            ]);

            if (aiResponse && aiResponse.trim() !== '') {
                return res.json({ reply: aiResponse, source: 'RAG Engine + Groq AI' });
            }

            const safeFallback = 'Mohon maaf, saya belum menemukan aturan spesifik mengenai hal tersebut di buku pedoman akademik saat ini. Bisa tolong berikan kata kunci yang lebih jelas?';
            recordFallbackTurn(activeUserId, message, safeFallback);
            return res.json({ reply: safeFallback, source: 'Safe Fallback' });
        } catch (aiError) {
            console.error('AI Processing Error:', aiError.message);
            const timeoutFallback = 'Maaf, sistem AI sedang mengalami antrean komputasi. Silakan coba kirimkan ulang pertanyaan Anda.';
            return res.json({ reply: timeoutFallback, source: 'Timeout Fallback' });
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
    const docs = datasetManager.getDatasetDocuments
        ? datasetManager.getDatasetDocuments(req.params.name)
        : [];
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
        if (!datasetManager.saveDataset) {
            return res.status(501).json({ message: 'saveDataset belum diimplementasikan di DatasetManager' });
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
    console.log('=========================================');
    console.log('Server BERJALAN murni berbasis Website.');
    console.log(`Akses Chat App: http://localhost:${PORT}`);
    console.log(`Akses Panel Admin: http://localhost:${PORT}/admin.html`);
    console.log('=========================================');

    const activeDocs = datasetManager.getAllDocuments();
    ragEngine.buildVectorIndex(activeDocs);
    console.log(`Datasets loaded: ${datasetManager.listDatasets().length} file CSV.`);
});
