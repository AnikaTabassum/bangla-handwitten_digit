import cv2


def getDigitsFrames(numberFrame):
    grayImage = cv2.cvtColor(numberFrame, cv2.COLOR_BGR2GRAY)
    img2 = grayImage.copy()

    # binarizing threshold is 100, greater than 100 is white
    cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY, grayImage)

    # cv2.imshow('converted image to grayscale', grayImage)
    # cv2.waitKey(0)

    black = []  # number of black points per column
    white = []  # number of black dots per line

    row = img2.shape[0]
    col = img2.shape[1]

    for i in range(col):
        blackCnt = 0
        whiteCnt = 0
        for j in range(row):
            if grayImage[j][i] == 0:
                blackCnt += 1
            else:
                whiteCnt += 1
        black.append(blackCnt)
        white.append(whiteCnt)

    blackMax = max(black)  # maximum number of black spots
    whiteMax = max(white)  # maximum number of white spots

    is_black_background = whiteMax < blackMax
    # is the background color, no characters
    if (sum(black) if is_black_background else sum(white)) == row * col:
        print('no char')

    # Remove the white space at the top and bottom of the character
    def find_end(pixelRow, pixelCol):
        black = []
        for i in range(pixelRow):
            blackCnt = 0
            for j in range(pixelCol):
                if grayImage[i][j] == 0:
                    blackCnt += 1
            black.append(blackCnt)
        start_row = 0  # character start line
        end_row = pixelRow  # character end line

        for i in range(pixelRow):
            # This line is all the background
            if (black[i] if is_black_background else (pixelCol - black[i]) == pixelCol):
                continue
            else:
                start_row = i
            break
        for i in range(start_row + 1, pixelRow):
            # This line is all the background
            if (black[i] if is_black_background else (pixelCol - black[i]) == pixelCol):
                end_row = i
                break
        return start_row, end_row

    def find_end2(start_col, col):
        end_col = start_col

        # Found the beginning of the character column
        temp = -1
        for i in range(start_col, col - 1):
            # if (white[i] if is_black_background else black[i]) > (0.1 * (whiteMax if is_black_background else
            # blackMax)):
            if (white[i] if is_black_background else black[i]) > 0:
                temp = i
                print('start:', i)
                break
        if temp == -1:
            return start_col, end_col
        else:
            start_col = temp

        # Find the end of the character column
        for i in range(start_col + 1, col - 1):
            #  occupies 95% of the color of this column ---" End of the split character
            if (black[i] if is_black_background else white[i]) > (
                    0.9 * (blackMax if is_black_background else whiteMax)):
                # Next column also font color
                # if (white[i + 1] if is_black_background else black[i + 1]) > (
                #         0.1 * (whiteMax if is_black_background else blackMax)):
                if (white[i + 1] if is_black_background else black[i + 1]) > 0:
                    continue
                end_col = i
                print('end:', i)
                break
        return start_col, end_col

    start = 0
    end = 0
    top, bottom = find_end(row, col)
    # imggg = grayImage[top:bottom, 0:col]
    # cv2.imshow('1', imggg)
    # cv2.waitKey(0)
    i = 0
    digitFrames = []
    while i < col - 1:
        start, end = find_end2(start, col)
        i = end
        # print("ok")
        if start == end:  # No characters found
            break
        if end - start > 5:
            imggg = grayImage[top:bottom, start:end]
            digitFrames.append(imggg)
            # cv2.imshow('1', imggg)
            # # cv2.waitKey(2000)
            # fileName = 'resources/temp/image' + str(i) + '.jpg'
            # cv2.imwrite(fileName, imggg)
        # else:
        #     print("error")
        start = end + 1

    return digitFrames


img = cv2.imread("resources/temp/test.jpg")
digitsFrames = getDigitsFrames(img)
i = 0
for frame in digitsFrames:
    fileName = 'resources/temp/image' + str(i) + '.jpg'
    cv2.imwrite(fileName, frame)
    i += 1
