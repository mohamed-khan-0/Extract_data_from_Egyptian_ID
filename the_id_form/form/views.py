from django.shortcuts import render
from .forms import IDCardForm
from .main import front_text_recognition, back_text_recognition

def upload_id_card(request):
    if request.method == 'POST':
        form = IDCardForm(request.POST, request.FILES)
        if form.is_valid():
            front_image = request.FILES['front_image']
            back_image = request.FILES['back_image']

            front_image_path = 'front_image.jpg'
            back_image_path = 'back_image.jpg'

            with open(front_image_path, 'wb+') as destination:
                for chunk in front_image.chunks():
                    destination.write(chunk)

            with open(back_image_path, 'wb+') as destination:
                for chunk in back_image.chunks():
                    destination.write(chunk)

            # Extract data from images
            front_data = front_text_recognition(front_image_path)
            back_data = back_text_recognition(back_image_path)
            front_data.update(back_data)

            # Populate form with the extracted data
            form = IDCardForm(initial=front_data)

            context = {
                'form': form,
            }

            return render(request, 'id_card_form.html', context)
    else:
        form = IDCardForm()

    return render(request, 'id_card_form.html', {'form': form})
