# Aequa: Fair Model Rewards in Collaborative Learning via Slimmable Networks [ICML 2025]

[![Paper](https://img.shields.io/badge/OpenReview-Paper-f31f1f)](https://openreview.net/forum?id=Tw81RElDpe)
[![Paper](https://img.shields.io/badge/arXiv-Paper-teal)](https://arxiv.org/abs/2502.04850)

> [**Aequa: Fair Model Rewards in Collaborative Learning via Slimmable Networks [ICML 2025]**](https://arxiv.org/abs/2502.04850)<br>
> [Nurbek Tastan](https://tnurbek.github.io/), [Samuel Horvath](https://sites.google.com/view/samuelhorvath/home), [Karthik Nandakumar](https://cse.msu.edu/~nandakum/)



## Dependencies

```bash
pip install -r requirements.txt
```

## Run Aequa

Default dataset is CIFAR-10 with custom CNN. 

```python
python3 main_aequa.py -T 100 -S homogeneous -model_arch cnn -lr 0.01 
```

To test (to obtain width-wise performance measures), simply run `main_test.py` file with the same configurations from the training above: 
```python
python3 main_test.py -T 100 -S homogeneous -model_arch cnn -lr 0.01 
```


## Citation 
If you like the work, please consider citing us and explore other works: 

```bibtex
@inproceedings{tastan2025aequa,
  title     = {{Aequa: Fair Model Rewards in Collaborative Learning via Slimmable Networks}},
  author    = {Tastan, Nurbek and Horv\'{a}th, Samuel and Nandakumar, Karthik},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages     = {59210--59236},
  year      = {2025},
  editor    = {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume    = {267},
  series    = {Proceedings of Machine Learning Research},
  month     = {13--19 Jul},
  publisher = {PMLR},
  pdf       = {https://raw.githubusercontent.com/mlresearch/v267/main/assets/tastan25a/tastan25a.pdf},
  url       = {https://proceedings.mlr.press/v267/tastan25a.html}
}

@article{tastan2025cycle,
  title     = {{{CYC}le: Choosing Your Collaborators Wisely to Enhance Collaborative Fairness in Decentralized Learning}},
  author    = {Nurbek Tastan and Samuel Horv{\'a}th and Karthik Nandakumar},
  journal   = {Transactions on Machine Learning Research},
  issn      = {2835-8856},
  year      = {2025},
  url       = {https://openreview.net/forum?id=ygqNiLQqfH},
  note      = {}
}

@inproceedings{tastan2024redefining,
  title     = {{Redefining Contributions: Shapley-Driven Federated Learning}},
  author    = {Tastan, Nurbek and Fares, Samar and Aremu, Toluwani and Horváth, Samuel and Nandakumar, Karthik},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson}, 
  pages     = {5009--5017},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
}
```

# Acknowledgements 
We would like to thank [IAFL repository](https://github.com/ZhaoxuanWu/Incentive-Aware-FL) for open-sourcing their code. 
