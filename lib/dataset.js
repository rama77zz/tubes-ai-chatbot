const fs = require('fs');
const path = require('path');
const xlsx = require('xlsx');

class DatasetManager {
    constructor() {
        // Mengarah langsung ke nama file baru Anda di dalam folder data
        this.csvFilePath = path.join(__dirname, '../data/Dataset_Pedoman_Akademik_Telkom_2024.csv');
    }

    /**
     * Membaca file CSV mentah (Page & Text) dan memotongnya menjadi chunks untuk RAG
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

            // 3. Iterasi setiap halaman PDF yang ada di dalam CSV
            rawData.forEach((row) => {
                const pageText = row.Text || row.text || "";
                const pageNum = row.Page || row.page || "Unknown";

                if (!pageText.trim()) return;

                // Membersihkan spasi berlebih dan carriage return (\r)
                const cleanText = pageText.replace(/\r/g, '').replace(/\s+/g, ' ').trim();

                // Karena 1 halaman terlalu panjang, kita potong manual per 800 karakter
                const maxChunkSize = 800;
                let start = 0;

                while (start < cleanText.length) {
                    let end = start + maxChunkSize;
                    
                    // Usahakan memotong tepat pada spasi agar kata tidak terputus di tengah
                    if (end < cleanText.length) {
                        const nextSpace = cleanText.lastIndexOf(' ', end);
                        if (nextSpace > start + 200) {
                            end = nextSpace;
                        }
                    }

                    const chunkText = cleanText.slice(start, end).trim();

                    if (chunkText.length > 30) {
                        // Kategorisasi Cerdas Berdasarkan Kata Kunci Judul Halaman / Isi Konten
                        let kategori = "Akademik Umum";
                        const textLower = chunkText.toLowerCase();
                        
                        if (anyIncluded(textLower, ["cuti", "registrasi", "nonaktif", "undur", "pindah", "kartu", "perwalian", "krs", "ksm"])) {
                            kategori = "Administrasi & Layanan";
                        } else if (anyIncluded(textLower, ["lulus", "yudisium", "gelar", "ijazah", "skpi", "predikat"])) {
                            kategori = "Kelulusan & Tugas Akhir";
                        } else if (anyIncluded(textLower, ["nilai", "evaluasi", "sks", "indeks", "prestasi", "standar penilaian"])) {
                            kategori = "Evaluasi & Penilaian";
                        } else if (anyIncluded(textLower, ["magang", "rpl", "fast track", "internasional", "pjj", "jarak jauh", "wrap"])) {
                            kategori = "Program Khusus";
                        }

                        // Masukkan ke format penampung dokumen yang siap dibaca RAGEngine
                        formattedDocs.push({
                            pageContent: chunkText,
                            metadata: {
                                id: `page-${pageNum}-chunk-${start}`,
                                kategori: kategori,
                                topik: `Halaman ${pageNum}`,
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

// Fungsi pembantu untuk mencocokkan kata kunci kategori (Sudah Diperbaiki)
function anyIncluded(targetText, keywordsArray) {
    return keywordsArray.some(word => targetText.includes(word));
}

module.exports = DatasetManager;