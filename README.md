## From EHRs to Patient Pathways: Scalable Modeling of Longitudinal Health Trajectories with LLMs
**Authors:** [Chantal Pellegrini][cp], [Ege Özsoy][eo], [David Bani-Harouni][db], [Matthias Keicher][mk], [Nassir Navab][nn]

[cp]:https://www.cs.cit.tum.de/camp/members/chantal-pellegrini/
[eo]:https://www.cs.cit.tum.de/camp/members/ege-oezsoy/
[db]:https://www.cs.cit.tum.de/camp/members/david-bani-harouni/
[mk]:https://www.cs.cit.tum.de/camp/members/matthias-keicher/
[nn]:https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/

[![](https://img.shields.io/badge/Arxiv-2307.05766-blue)](TODO)

<img align="right" src="figs/ehr2path.pdf" alt="teaser" width="50%" style="margin-left: 20px">

Healthcare systems face significant challenges in managing and interpreting vast, heterogeneous patient data for personalized care. Existing approaches often focus on narrow use cases with a limited feature space, overlooking the complex, longitudinal interactions needed for a holistic understanding of patient health. In this work, we propose a novel approach to patient pathway modeling by transforming diverse electronic health record (EHR) data into a structured representation and designing a holistic pathway prediction model, EHR2Path, optimized to predict future health trajectories. Further, we introduce a novel summary mechanism that embeds long-term temporal context into topic-specific summary tokens, improving performance over text-only models, while being much more token-efficient. EHR2Path demonstrates strong performance in both next time-step prediction and longitudinal simulation, outperforming competitive baselines. It enables detailed simulations of patient trajectories, inherently targeting diverse evaluation tasks, such as forecasting vital signs, lab test results, or length-of-stay, opening a path towards predictive and personalized healthcare.

## Instructions
For detailed instructions on how to set up the environment, prepare the data, and train the models, please refer to the following readme files:
- [Setup](Readme_Setup.md)
- [Dataset Generation](Readme_Dataset_Generation.md)
- [Pathway Model Training and Evaluation](Readme_Pathway_Models.md)
- [Simulation Task Fine-tuning](Readme_Finetuning.md)

## Reference

```
@inproceedings{todo
}
```