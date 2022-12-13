import numpy as np
import cv2 
import argparse

def scale_im(im, scale):
    width = int(im.shape[1] * scale / 100)
    height = int(im.shape[0] * scale / 100)
    dim = (width, height)

    # resize image
    im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    return im

def feat(infile, outfile, scale):
    img = cv2.imread(infile)
    gray= scale_im( cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), scale )

    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)

    img=cv2.drawKeypoints(gray,kp,img)
    cv2.imwrite(outfile,img)

def match(infile1, infile2, outfile, scale):
    img1 = scale_im( cv2.imread(infile1,cv2.IMREAD_GRAYSCALE), scale )
    img2 = scale_im( cv2.imread(infile2,cv2.IMREAD_GRAYSCALE), scale )
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(outfile,img3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure from Motion")
    parser.add_argument("-i1", "--infile1", type=str, required=True, help="Image to run on.")
    parser.add_argument("-i2", "--infile2", type=str, help="Other image to run file on.")
    parser.add_argument("-t", "--type", type=str, required=True, help="What figure to make. One of feat, match")
    parser.add_argument("-s", "--scale", type=float, default=100, help="What figure to make. One of feat, match")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="Output file to save to")
    args = parser.parse_args()
    
    if args.type == "feat":
        feat(args.infile1, args.outfile, args.scale)
    elif args.type == "match":
        match(args.infile1, args.infile2, args.outfile, args.scale)