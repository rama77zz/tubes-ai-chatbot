const fs = require('fs');
const path = require('path');
const xlsx = require('xlsx');

class DatasetManager {
    constructor() {
        // Mengarah langsung ke file dataset baru Anda
        this.csvFilePath = path.join(__dirname, '../data/Dataset_Pedoman_Akademik_Telkom_2024.csv');
    }

    /**
     * Membaca file CSV mentah berdasarkan struktur kolom baru (content, kategori, topik)
     * @returns {Array} List data berisi objek { pageContent, metadata }
     */
    getAllDocuments() {
        try {
            if (!fs.existsSync(this.csvFilePath)) {
                console.error(`[Dataset Error] File tidak ditemukan di: ${this.csvFilePath}`);
                return [];
            }

            // 1. Baca berkas CSV
            const workbook = xlsx.readFile(this.csvFilePath);
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];

            // 2. Konversi worksheet menjadi JSON Array
            const rawData = xlsx.utils.sheet_to_json(worksheet);
            const formattedDocs = [];

            // 3. Iterasi setiap baris pedoman akademik
            rawData.forEach((row, index) => {
                // SESUAIKAN DENGAN NAMA KOLOM BARU DI CSV ANDA: 'content', 'kategori', 'topik'
                const mainText = row.content || row.Content || "";
                const currentTopic = row.topik || row.topik || `Aturan-${index}`;
                const currentCategory = row.kategori || row.kategori || "Umum";

                if (!mainText.trim()) return;

                // Membersihkan spasi berlebih dan carriage return (\r)
                const cleanText = mainText.replace(/\r/g, '').replace(/\s+/g, ' ').trim();

                // Karena baris teks di dataset baru sudah cukup padat, kita potong manual per 800 karakter jika terlalu panjang
                const maxChunkSize = 800;
                let start = 0;

                while (start < cleanText.length) {
                    let end = start + maxChunkSize;
                    
                    if (end < cleanText.length) {
                        const nextSpace = cleanText.lastIndexOf(' ', end);
                        if (nextSpace > start + 200) {
                            end = nextSpace;
                        }
                    }

                    const chunkText = cleanText.slice(start, end).trim();

                    if (chunkText.length > 20) {
                        // Masukkan ke format penampung dokumen yang siap dibaca RAGEngine
                        formattedDocs.push({
                            pageContent: chunkText,
                            metadata: {
                                id: `row-${index}-chunk-${start}`,
                                kategori: currentCategory,
                                topik: currentTopic,
                                sumber: "Pedoman Akademik TUS 2024"
                            }
                        });
                    }
                    start = end + 1;
                }
            });

            return formattedDocs;
        } catch (error) {
            console.error('Error saat mengekstrak dataset CSV mentah:', error.message);
            return [];
        }
    }

    listDatasets() {
        try {
            if (fs.existsSync(this.csvFilePath)) {
                return [path.basename(this.csvFilePath)];
            }
            return [];
        } catch (error) {
            return [];
        }
    }
}

module.exports = DatasetManager;