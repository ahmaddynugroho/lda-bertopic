#### start

```plantuml
start
:Studi pustaka;
:Mempersiapkan\nenvironment\npython;
:Pengambilan data\n(web scraping);
:Preprocessing;
:Pemodelan topik;
:Evaluasi dan komparasi;
stop
```

gambar 3.1 langkah-langkah penelitian

```plantuml
start
:Menyiapkan URL\nartikel berita;
:Mencari tag HTML\nyang mengandung\nartikel;
:Membuat script scraping;
:Menjalankan script\nscraping;
:Menyimpan data\nke dalam format csv;
stop
```

gambar 3.2 langkah-langkah web scraping

```plantuml
package "kalimat/dok" {
    [LDA]
    [BERTopic]
}
```

```plantuml
start
:Pelatihan model topik menggunakan algoritma BERTopic;
:Pelatihan model topik menggunakan algoritma LDA
dengan jumlah topik yang dihasilkan oleh BERTopic;
:Menyimpan hasil ke dalam format csv;
stop
```

```plantuml
collections corpus
participant Bertopic as bert
participant LDA as lda
control Evaluasi as eval
database CSV as csv
corpus  -> bert : pelatihan model
bert -> eval : evaluasi topic coherence,\ntopic diversity, dan waktu
corpus -> lda : pelatihan model
lda -> eval : evaluasi topic coherence,\ntopic diversity, dan waktu
eval -> csv : menyimpan hasil\nevaluasi
```
