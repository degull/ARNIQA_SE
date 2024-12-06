# ARNIQA (WACV 2024 Oral)

### Learning Distortion Manifold for Image Quality Assessment

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.14918)
[![Generic badge](https://img.shields.io/badge/Video-YouTube-red.svg)](https://youtu.be/UUwpoi61jpg)
[![Generic badge](https://img.shields.io/badge/Slides-Link-orange.svg)](/assets/Slides.pptx)
[![Generic badge](https://img.shields.io/badge/Poster-Link-purple.svg)](/assets/Poster.pdf)
[![GitHub Stars](https://img.shields.io/github/stars/miccunifi/ARNIQA?style=social)](https://github.com/miccunifi/ARNIQA)

## Citation

```bibtex
@inproceedings{agnolucci2024arniqa,
  title={ARNIQA: Learning Distortion Manifold for Image Quality Assessment},
  author={Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={189--198},
  year={2024}
}
```


## Authors

* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)
* [**Leonardo Galteri**](https://scholar.google.com/citations?user=_n2R2bUAAAAJ&hl=en)
* [**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.com/citations?user=bf2ZrFcAAAAJ&hl=en)

## Acknowledgements

This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number 101004545 - ReInHerit.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.



### SE-ResNet50
- Bottleneck 블록에 SEBlock 추가 → Bottleneck 마지막 단계에서 활성화 돼 출력에 적용
- 채널별 가중치를 학습해 중요도 조정
- 입력 특징 맵 x에 대해 Global Average Pooling, FC Layer, Sigmoid를 거쳐 채널별 가중치를 계산하고 입력 특징 맵에 곱
- 채널 간 중요도를 학습해 입력 특징의 가중치 조정
- ResNet50의 모든 Bottleneck 블록을 SE-enabled Bottleneck으로 대체
- Feature Extraction과 Projection Head를 분리하여 임베딩 벡터 생성


### 기존 ARN-IQA와의 차이점

| **구분**               | **기존 ARN-IQA**                           | **현재 코드**                                   |
|------------------------|-------------------------------------------|-----------------------------------------------|
| **Backbone**           | ResNet50                                 | SE 블록이 포함된 ResNet50 (`ResNetSE`)        |
| **SE 블록**            | 없음                                      | Bottleneck 블록에 추가된 SE 블록              |
| **Projection Head**    | SimCLR 방식의 Projection Head             | 비슷한 구조이나, SE 블록과 결합된 형태        |
| **Feature Extraction** | ResNet50의 Feature Extraction 사용       | SE 블록을 통해 채널별 중요도 조정된 특징 맵   |
| **Fully Connected**    | ResNet50의 원래 FC 제거 후 SimCLR Head    | ResNet50의 원래 FC 제거 및 Projector 추가     |

---



