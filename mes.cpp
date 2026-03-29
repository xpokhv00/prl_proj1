/**
* PRL 2026 Projekt 1
* Autor: Vsevolod Pokhvalenko <xpokhv00>
*/

#include <mpi.h>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Candidate {
    int value; // 0..255 nebo sentinel
    int index; // globální index
};

// Candidate comparator: vrátí "lepší" kandidát podle (value, index).
// Nižší value je lepší; při shodě vyhraje menší původní globální index
static inline Candidate better(const Candidate& a, const Candidate& b) {
    if (a.value < b.value) return a;                 // a má menší hodnotu
    if (b.value < a.value) return b;                 // b má menší hodnotu
    return (a.index <= b.index) ? a : b;             // stejné hodnoty -> nižší index vyhrává
}

// Načte celý binární soubor do vektoru bajtů.
// Vrací true při úspěchu, false při chybě.
static bool read_all_bytes(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    // zjistíme velikost souboru
    f.seekg(0, std::ios::end);
    std::streampos sz = f.tellg();
    if (sz < 0) return false;
    f.seekg(0, std::ios::beg);

    // alokujeme výsledný vektor a přečteme obsah
    out.resize(static_cast<size_t>(sz));
    if (!out.empty()) {
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
        if (!f) return false;
    }
    return true;
}

// Rozdělení N prvků mezi P procesů pro MPI_Scatterv.
// counts[r] = počet prvků pro rank r, displs[r] = offset (globální index) pro rank r.
// Rozdělíme rovnoměrně: první (N % P) procesů dostane o 1 více.
static void make_scatterv_layout(int N, int P, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(P, 0);
    displs.assign(P, 0);

    const int base = (P == 0) ? 0 : (N / P); // základní počet na proces
    const int rem  = (P == 0) ? 0 : (N % P); // zbytek, který rozdělíme po jednom

    int offset = 0;
    for (int r = 0; r < P; ++r) {
        counts[r] = base + (r < rem ? 1 : 0); // první rem procesů dostane +1
        displs[r] = offset;                   // počáteční index pro tento proces
        offset += counts[r];
    }
}

// Najde lokální minimum v poli local.
// global_offset je počáteční globální index prvku local[0].
// Vrací Candidate s (value, global_index) lokálního minima.
static Candidate local_min_with_index(const std::vector<int>& local, int global_offset, int sentinel) {
    Candidate c;
    c.value = sentinel;
    c.index = std::numeric_limits<int>::max();

    for (int i = 0; i < static_cast<int>(local.size()); ++i) {
        const int v = local[i];
        const int idx = global_offset + i;
        Candidate cur{v, idx};
        c = better(c, cur);
    }

    return c;
}

// Jedna stromová redukce pro (value,index) do rank 0.
// Všichni procesy projdou stejnými kroky; ti, co posílají, v redukci končí.
static Candidate tree_reduce_minloc(Candidate cand, int rank, int size) {
    for (int step = 1; step < size; step <<= 1) {
        if ((rank & step) != 0) {
            int partner = rank - step;
            if (partner >= 0) {
                int buf[2] = { cand.value, cand.index };
                MPI_Send(buf, 2, MPI_INT, partner, 0, MPI_COMM_WORLD);
            }
            break; // posílající proces už dál nepokračuje
        } else {
            int partner = rank + step;
            if (partner < size) {
                int buf[2];
                MPI_Recv(buf, 2, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Candidate other{buf[0], buf[1]};
                cand = better(cand, other);
            }
        }
    }
    return cand; // validní globální minimum je na rank 0
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<unsigned char> global_bytes;
    int N = 0;

    // Sentinel musí být > 255, aby nikdy nevyhrál jako minimum.
    const int SENTINEL = 256;

    // 1) Rank 0 načte soubor a vypíše vstup
    if (rank == 0) {
        if (!read_all_bytes("numbers", global_bytes)) {
            std::cerr << "Chyba: nelze nacist soubor 'numbers' (errno=" << errno << ").\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N = static_cast<int>(global_bytes.size());
        if (N <= 0) {
            std::cerr << "Chyba: soubor 'numbers' je prazdny.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Výpis vstupní posloupnosti (oddělena jednou mezerou, + mezera před newline dle ukázky)
        for (int i = 0; i < N; ++i) {
            std::cout << static_cast<int>(global_bytes[i]) << (i + 1 == N ? " \n" : " ");
        }
    }

    // 2) Broadcast N všem
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 3) Scatterv rozdělení dat
    std::vector<int> counts, displs;
    make_scatterv_layout(N, size, counts, displs);

    const int local_n = counts[rank];
    std::vector<unsigned char> local_bytes(static_cast<size_t>(local_n));

    MPI_Scatterv(
        rank == 0 ? global_bytes.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_UNSIGNED_CHAR,
        local_bytes.data(),
        local_n,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    // převedeme na int, protože budeme používat SENTINEL=256
    std::vector<int> local(static_cast<size_t>(local_n), SENTINEL);
    for (int i = 0; i < local_n; ++i) local[i] = static_cast<int>(local_bytes[i]);

    const int global_offset = displs[rank];

    // 4) Minimum extraction sort: N-krát najdi globální minimum, vypiš, a "odstraň"
    for (int iter = 0; iter < N; ++iter) {
        // lokální kandidát
        Candidate cand = local_min_with_index(local, global_offset, SENTINEL);

        // stromová redukce na rank 0
        Candidate reduced = tree_reduce_minloc(cand, rank, size);

        int gbuf[2];
        if (rank == 0) {
            gbuf[0] = reduced.value;
            gbuf[1] = reduced.index;
        }

        // Broadcast globálního kandidáta (value,index) z rank 0 všem, aby věděli, co odstranit
        MPI_Bcast(gbuf, 2, MPI_INT, 0, MPI_COMM_WORLD);

        // Všichni teď mají globální minimum (value,index) a ten, kdo ho vlastní, ho odstraní
        Candidate global_cand{gbuf[0], gbuf[1]};

        // Rank 0 vypíše hodnotu minima
        if (rank == 0) {
            std::cout << global_cand.value << "\n";
        }

        // Ten, kdo vlastní globální minimum, ho nahradí SENTINELem
        const int gidx = global_cand.index;
        if (gidx >= global_offset && gidx < global_offset + local_n) {
            local[gidx - global_offset] = SENTINEL;
        }

        // Synchronizace před další iterací
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Ukončení exekučního prostředí MPI
    MPI_Finalize();

    // Ukončení programu
    return 0;
}