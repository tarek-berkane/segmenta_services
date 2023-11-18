from django.views import generic
from django.views.decorators.http import require_POST
from django.http import HttpResponse, HttpResponseBadRequest
from src.pipeline import predict_pipeline


from ml.models import FileModel
from ml.pipline import customer_pipeline


@require_POST
def upload_file(request):
    if request.FILES:
        file = request.FILES["file"]
        file_obj = FileModel.objects.create(file=file)
        return HttpResponse(file_obj.id)
    return HttpResponseBadRequest()


# @require_POST
# def run_forcasting(request):
#     print(request.POST)
#     if file_id := request.POST.get("file"):
#     return (reverse("website:home"))


class SalesForecastView(generic.TemplateView):
    template_name = "sales_forecasting/index.html"

    def post(self, request, *args, **kwargs):
        file_id = request.POST.get("file")
        if not file_id:
            return HttpResponseBadRequest()
        file_model = FileModel.objects.get(id=file_id)
        print(file_model.file.path)
        forcast_data = predict_pipeline.predict(file_model.file.path)
        context = self.get_context_data()
        context["data"] = forcast_data.to_dict(index=True)
        return self.render_to_response(context)


class CustomerSegmentationView(generic.TemplateView):
    template_name = "customer_segmentation/index.html"

    def post(self, request, *args, **kwargs):
        file_id = request.POST.get("file")
        if not file_id:
            return HttpResponseBadRequest()
        file_model = FileModel.objects.get(id=file_id)
        forcast_data = customer_pipeline.predict(file_model.file.path)
        data = {}
        print(forcast_data)
        counter = 0
        for index, item in forcast_data.iterrows():
            # print(item)
            counter += 1
            if counter > 800:
                break
            if not data.get(item["cluster_pca"]):
                data[item["cluster_pca"]] = []
            data[item["cluster_pca"]].append(
                [round(item["PURCHASES"], 2), round(item["PAYMENTS"], 2)]
            )

        context = self.get_context_data()
        context["data"] = data
        return self.render_to_response(context)
