# AI Genomic Prediction Pipeline

Nextflow pipeline for genomic prediction using Huggingface models & trainer on public maize genomic data.

## Quick Start
```bash
# 1. Clone repository
git clone https://github.com/karlenenegus/AI_GS.git
cd AI_GS

# 2. Download public data (~500MB, takes 2-5 minutes)
bash data/download_data.sh

# 3. Setup environment
conda env create -f environment.yml
conda activate AI_GS

# 4. Test pipeline with tiny data (30 seconds)
nextflow run main.nf \
    --geno_file test/test_tiny_data/tiny_geno_10snps.hmp \
    --pheno_file test/test_tiny_data/tiny_pheno_10lines.csv \
    -profile test

# 5. Run full pipeline (4 hours on HPC)
nextflow run main.nf -profile slurm
```

## Data

This pipeline uses **publicly available data** from Tian et al. (2023):
- 12,000 SNP markers from 500 maize inbred lines
- Grain yield phenotypes across 5 environments
- All data downloaded automatically via script

**No data is stored in this repository** - data is downloaded on-demand.

See [data/README.md](data/README.md) for details.

## 📁 Repository Contents
```
├── main.nf              # Main pipeline (all 6 analysis steps)
├── modules/             # Reusable process modules
├── bin/                 # Python/R analysis scripts
├── data/                # Data directory (empty - run download script)
├── test/                # Tiny test data (committed to git)
└── docs/                # Documentation
```

## 🎯 What This Repository Shows

For **hiring managers** and **collaborators**, this demonstrates:

✅ **Real-world bioinformatics**: Working pipeline on published data  
✅ **Reproducible science**: One command reproduces entire analysis  
✅ **Software engineering**: Modular design, documentation, testing  
✅ **HPC expertise**: Efficient resource management, SLURM integration  
✅ **Open practices**: Public data, shared code, clear documentation

See [docs/FOR_HIRING_MANAGERS.md](docs/FOR_HIRING_MANAGERS.md) for technical details.

## 📖 Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Data Sources](data/README.md)
- [Pipeline Overview](docs/pipeline_overview.md)
- [For Hiring Managers](docs/FOR_HIRING_MANAGERS.md)

## 🧪 Testing
```bash
# Quick test with tiny data (included in repo)
nextflow run main.nf \
    --geno_file test/test_tiny_data/tiny_geno_10snps.hmp \
    --pheno_file test/test_tiny_data/tiny_pheno_10lines.csv \
    -profile test
# Expected runtime: 30 seconds
```

## 🔧 Requirements

- Nextflow ≥21.04.0
- Conda or Docker
- ~1GB disk space for data
- SLURM cluster (optional - can run locally)

## 📄 Citation

Code:
```bibtex
@software{your_pipeline_2025,
  author = {Your Name},
  title = {Genomic Prediction Pipeline},
  year = {2025},
  url = {https://github.com/yourusername/genomics-pipeline}
}
```

Data source:
```bibtex
@article{tian2023genomic,
  title={Genomic prediction in maize hybrids},
  author={Tian, F. and others},
  journal={Crop Science},
  year={2023}
}
```

## 📝 License

MIT License - See [LICENSE](LICENSE)

Data: CC BY 4.0 (from original publication)

## 📧 Contact

[Your Name] - you@email.com  
Project: https://github.com/yourusername/genomics-pipeline
