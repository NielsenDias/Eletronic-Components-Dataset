directory = 'treated_data'
new_directory = 'resampled_data'
img_size = 100

if not os.path.exists(new_directory):
    os.makedirs(new_directory)

for folder in os.listdir(directory):
    for component in os.listdir(f'{directory}/{folder}'):
        img_names =  os.listdir(f'{directory}/{folder}/{component}')
        for image_id, image_file in enumerate(tqdm(img_names,desc=folder+component)):
            current_imgs = [cv2.imread(f'{directory}/{folder}/{component}/{image_file}')]
            
            mirror_imgs = []
            for img in current_imgs:
                mirror_imgs.append(numpy.fliplr(img))
                mirror_imgs.append(numpy.flipud(img))
            current_imgs += mirror_imgs
            del mirror_imgs
            
            shift_amounts = [((random.randint(15,60),random.randint(15,60)),(random.randint(15,60),random.randint(15,60)),(0,0)) for i in range(3)]
            shift_imgs = []
            for img in current_imgs:
                for shift_amount in shift_amounts:
                    shift_imgs.append(numpy.pad(img,shift_amount,mode='constant', constant_values=255))
            current_imgs += shift_imgs
            del shift_imgs

            # blur_amounts = [random.randint(0,5) for i in range(4)]
            blur_amounts = [3]
            blur_imgs = []
            for img in current_imgs:
                for blur_amount in blur_amounts:
                    blur_imgs.append(ndimage.gaussian_filter(img, sigma=blur_amount))
            current_imgs += blur_imgs
            del blur_imgs

            # rotate_amounts = [random.randint(0,180) for i in range(2)]
            rotate_amounts = [45,90]
            rotate_imgs = []
            for img in current_imgs:
                for rotate_amount in rotate_amounts:
                    rotate_imgs.append(ndimage.rotate(img, rotate_amount, reshape=False,cval=255))
            current_imgs += rotate_imgs
            del rotate_imgs

            degrade_imgs = []
            for img in current_imgs:
                img_copy = copy.deepcopy(img)
                size = (min(img.shape[:2])-2)
                x, y = (size*numpy.random.random((2, 5000))).astype(int)
                img_copy[x, y] = 1
                degrade_imgs.append(img_copy)
            current_imgs += degrade_imgs
            del degrade_imgs

            if not os.path.exists(f'{new_directory}/{folder}/{component}'):
                os.makedirs(f'{new_directory}/{folder}/{component}')
            saved_imgs =  [int(i.split('.')[0]) for i in os.listdir(f'{new_directory}/{folder}/{component}')]
            if not saved_imgs:
                saved_imgs.append(0)
            for ind, img in enumerate(current_imgs):
                img_array = cv2.resize(img, (img_size, img_size))
                cv2.imwrite(f'{new_directory}/{folder}/{component}/{ind+max(saved_imgs)+1}.jpg', img_array)

            del current_imgs

            gc.collect()