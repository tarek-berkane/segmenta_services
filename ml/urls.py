from django.urls import path
import ml.views as ml_views
import ml.api as ml_api

app_name = "website"

urlpatterns = [
    path("upload-file/", ml_views.upload_file, name="upload-file"),
    path(
        "sales-forecast/", ml_views.SalesForecastView.as_view(), name="sales-forecast"
    ),
    path(
        "customer-segmentation/",
        ml_views.CustomerSegmentationView.as_view(),
        name="customer-segmentation",
    ),
    # =============
    # API
    # =============
    path(
        "api/sales-forecast/",
        ml_api.SalesForecastAPI.as_view(),
        name="api-sales-forecast",
    ),
    path(
        "api/customer-segmentation/",
        ml_api.CustomerSegmentationAPI.as_view(),
        name="api-sales-forecast",
    ),
]
