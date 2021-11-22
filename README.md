# Тестовое задание
Требуется обучить модель кластеризации, основанную на подходе автоэнкодеров и измерить качество кластеризации, выдаваемое этой моделью.


### Рекомендуемые шаги для выполнения задания  
**Шаг 1.** Ознакомиться со [статьёй](https://arxiv.org/pdf/1806.10069.pdf), описывающей пользу применения совместной оптимизации для задачи кластеризации.  

В статье описывается польза применения совместной оптимизации по сравнению с последовательной. Предлагается использовать автоэнкодер для получения эмбеддингов текстов, что даёт лосс реконструкции. Необходимость решения задачи кластеризации приводит к использованию лосса кластеризации. Таким образом необходимо минимизировать оба лосса совместно вместо того, чтобы сначала обучить модель для генерации эмбеддингов, а затем с помощью этих эмбеддингов решать задачу кластеризации.

**Шаг 2.** Реализовать модель, описанную в стетье.

На данном шаге предлагается реализовать модель, описанную в статье из **Шага 1**.

**Шаг 3.** Скачать датасет Interfax.  

[Датасет](https://drive.google.com/drive/folders/1U8CZr5AgQHLfhgGiKSeg3Zd-tsXPP6i1?usp=sharing) состоит из заголовков русскоязычных новостей, относящихся к 15 категориям. Таким образом предстоит работать с кластеризацией коротких текстов. В колонке "text" содержится заголовок новости, в "topic" - категория новости.

Датасет уже поделён на train, val и test части. Предлагается обучить модель на train-части. Если необходимо - используйте val-часть. Итоговую оценку качества модели предлагается получить на test-части датасета.

**Шаг 4.** Обучить модель кластеризации на датасете Interfax.  

Обучить модель с двумя лоссами: reconstruction и clustring. Для обучения использовать датасет Interfax.

**Шаг 5.** Оценить качество обученной модели.  

NMI и ARI - классические метрики оценки качества решения задачи кластеризации. Предлагается получить метрики NMI и ARI для обученной модели кластеризации на test-части датасета.  

Также проверьте полученные кластеры на вырождаемость. Равно ли количество полученных кластеров числу классов в датесете? Если появились вырожденные кластеры, проанализируйте почему.

### Что будет оцениваться?
1. Оформление кода на github.
2. Оформление результатов.
3. Структура репозитория.
4. Соответствие решения тестовому заданию.
5. Любые релевантные теме мысли, идеи и соображения.

---

Резюме можно отправлять на почту info@machine-intelligence.ru или через форму на [сайте](http://machine-intelligence.ru/page11641715.html#Vacancy).
