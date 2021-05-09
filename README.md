# StylePTB: A Compositional Benchmark for Fine-grained Controllable Text Style Transfer


## Paper

[**StylePTB: A Compositional Benchmark for Fine-grained Controllable Text Style Transfer**](https://arxiv.org/abs/2104.05196)<br>
Yiwei Lyu*, [Paul Pu Liang*](http://www.cs.cmu.edu/~pliang/), Hai Pham*, [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/), and [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/)<br>
NAACL 2021. (*equal contribution)

If you find this repository useful, please cite our paper:
```
@article{lyu2021styleptb,
  title={StylePTB: A Compositional Benchmark for Fine-grained Controllable Text Style Transfer},
  author={Lyu, Yiwei and Liang, Paul Pu and Pham, Hai and Hovy, Eduard and P{\'o}czos, Barnab{\'a}s and Salakhutdinov, Ruslan and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2104.05196},
  year={2021}
}
```

----------------------------

To checkout single style transfers, use single_transform_checkout.py with the three letter style code as follows:

python single_transform_checkout.py [3-letter style code]

After you run the script, the data will be contained in a folder with the 3-letter code as name.

3 letter style codes:

TFU == To Future

TPA == To Past

TPR == To Present

ATP == Active To Passive

PTA == Passive To Active

PFB == PP Front To Back

PBF == PP Back To Front

IAD == Information Addition

ARR == ADJ/ADV Remooval

SBR == Substatement Removal

PPR == PP Removal

AEM == ADJ Emphasis

VEM == Verb Emphasis

NSR == Noun Synonym Replacement

ASR == Adjective Synonym Replacement

VSR == Verb Synonym Replacement

NAR == Noun Antonym Replacement

AAR == Adjective Antonym Replacement

VAR == Verb Antonym Replacement

LFS == Least Frequent Synonym Replacement

MFS == Most Frequent Synonym Replacement

----------------------------

To access the compositional datasets, a few of them are provided in the "Compositional Datasets" folder. No checkout needed.

----------------------------

The scripts for GPT baseline, GRU+attn, Retrieve-Edit, and StyleGPT are in the "Model Codes" folder. Note that the code for Retrieve-Edit is taken directly from the codes provided by the authors of "A Retrieve-and-Edit Framework for Predicting Structured Outputs" (NeurIPS2018)

---------------------------

The scripts used to perform automated transfers with parse trees are in the "Automatic Transfer Scripts", and the webpages and full results of the human annotated transfers are in "Amazon Mechanical Turk Webpages and Results" folder.
