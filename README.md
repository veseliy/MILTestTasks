# MILTestTask - OCR

## Result
Ноутбук для оценки результата: [/notebooks/base_model.ipynb](https://github.com/veseliy/MILTestTasks/blob/task/OCR_doc_layout/notebooks/base_model.ipynb)
окружение requirements.txt


## Задание
Мы предлагаем обучить модель для решения задачи Layout Detection на нашем [датасете](https://drive.google.com/file/d/1euOGyo8jzP-iJF_WMuwTtBzrRsvQ4h3c/view?usp=sharing).  
В архиве есть папка `data` с изображениями, и 2 json файла в [формате COCO для задачи detection](https://cocodataset.org/#format-data), c train и test частями соответственно. 
Данные для сегментации приведены в формате полигонов.
  
Для работы с форматом COCO рекомендуется использовать библиотеку `pycocotools`.

Код необходимый для получения результатов обучения модели нужно приложить в форке этого репозитория.  
Отчет по процессу решения и итоговым результатам желательно оформить как jupyter ноутбук с метриками mean IoU для тестовой и трейновой частей.

Перед решением рекомендуется взглянуть на датасет. 
В качестве базовых решений предлагаем ознакомиться со статьями [раз](https://arxiv.org/pdf/1512.02325.pdf) и [два](https://link.springer.com/chapter/10.1007/978-3-319-95957-3_30).
