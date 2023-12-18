from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import MGLMSummarizationPreprocessor
from modelscope.utils.constant import Tasks

model = 'ZhipuAI/Multilingual-GLM-Summarization-en'
preprocessor = MGLMSummarizationPreprocessor()
pipe = pipeline(
    task=Tasks.text_summarization,
    model=model,
    preprocessor=preprocessor,
    model_revision='v1.0.1',
)
result = pipe(
    '诊断规则配置页面有哪几个配置环节？'
)
print(result)