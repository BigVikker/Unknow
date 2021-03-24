<html>
<center>
 <strong>
  <h1>i use custom HaarCasscade to train my model at this address "https://www.cs.auckland.ac.nz/~m.rezaei/Tutorials/Creating_a_Cascade_of_Haar-Like_Classifiers_Step_by_Step.pdf"
  </h1>
 </strong>
</center>


<h3>1.Faces detect</h3>
- my model with 35 images positive, 400 images negative (i dont have data and my computer not strong to train)<br/>
- Evaluation model (work on data set :(( )
  <br/><br/>
  - With IOU = 0.5, set model scaleFactors(multiScale) = 1.01 (1%)
  <br/>
   ReCall: 65.7% , TP = 23, sum = 35
   <br/>
   Predection: 95.83%, TP = 23 and FP = 1
   <br/> <br />
  - With IOU = 0.5, set mode; scaleFactors(multiScale) = 1.05 (5%)
   <br/>
   ReCall: 40% , TP = 14, sum = 35
   <br/>
   Predection: 100%, TP = 14 and FP = 0
   <br/>


<h3>2. Number-Plate-motobike detect (private)</h3>
- my model with 200 positive, 400 images negative (special condition) <br/>
- Evaluation model (work on data set :(( )
<br/><br/>
 - With IOU = 0.5, set model scaleFactors(multiScale) = 1.01 (1%)<br/>
   ReCall: 73.68% , TP = 154 and sum = 209<br/>
   Predection: 96.85% , TP = 154 and FPS = 5<br/><br/>
 - With IOU = 0.5, set model scaleFactors(multiScale) = 1.05 (5%)<br/>
   ReCall: 29.19% , TP = 61 and sum = 209<br/>
   Predection: 95.31% , TP = 61 and FP = 3<br/>

<h3>3. Number-Plate-cars detect</h3>
- my model with 100 positive, 200 images negative (special condition) <br/>
- Evaluation model (work on data set :(( )
<br/><br/>
  - With IOU = 0.5, set model scaleFactors(multiScale) = 1.01 (1%)<br/>
   Recall: 50% , TP = 50 and sum = 100<br/>
   Predection: 13.97% , TP = 50 and FP = 308<br/><br/>
  - With IOU = 0.5, set model scaleFactors(multiScale) = 1.05 (5%)<br/>
   Recall: 51% , TP = 51 and sum = 100<br/>
   Predection: 20.9% , TP = 51 and FP = 163<br/><br/>
   
 Note: Result so bad cause by the conditions of images<br/>
 compare to result of haarcascade_russian_plate_number (on my data)<br/><br/>
 - With IOU = 0.5, set model scaleFactors(multiScale) = 1.01 (1%)<br/>
   Recall: 43% , TP = 44 and sum = 108<br/>
   Predection: 12.29% , TP = 44 and FP = 314<br/><br/>
 - With IOU = 0.5, set model scaleFactors(multiScale) = 1.05 (5%)<br/>
   Recall: 35% , TP = 38 and sum = 108<br/>
   Predection: 17.35% , TP = 38 and FP = 181<br/>
<h3> 4. Number-Plate-cars detect combine detect text by OCR </h3>
OCR-75% or 75% that mean OCR read correct 75%
OCR or 100% that mean OCR read correct 100%
<div>
 <center><h4> This show data Recall </h4>
<img src="data/graph.png" />
 </center>
</div>

<div>
<center><h4> This show data Recall of OCR via tesseract</h4>
<img src="data/graph2.png" /></center>
</div>

<div>
 <center>
  <h4> This show data Recall of special conditions and add more funtions in code</h4>
<img src="data/graph3.png" />
</center>
</div>

</html>
