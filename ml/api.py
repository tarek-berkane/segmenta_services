from django.views import View
from django.http import HttpRequest

from ml.models import FileModel
from ml.pipline import customer_pipeline
import src.pipeline.predict_pipeline as sales_pipeline
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


@method_decorator(csrf_exempt, name="dispatch")
class CustomerSegmentationAPI(View):
    def post(self, request: HttpRequest, *args, **kwargs):
        file = request.FILES["file"]
        file_model = FileModel.objects.create(file=file)
        forcast_data = customer_pipeline.predict(file_model.file.path)
        return JsonResponse(forcast_data.to_dict(index=True))


@method_decorator(csrf_exempt, name="dispatch")
class SalesForecastAPI(View):
    def post(self, request: HttpRequest, *args, **kwargs):
        file = request.FILES["file"]
        file_model = FileModel.objects.create(file=file)
        forcast_data = sales_pipeline.predict(file_model.file.path)
        return JsonResponse(forcast_data.to_dict(index=True))
