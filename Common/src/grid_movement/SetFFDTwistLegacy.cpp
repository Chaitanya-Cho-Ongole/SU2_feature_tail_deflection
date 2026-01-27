// NOTE: Copy this legacy (original SU2 version) incase the adaptive twist method fails

bool CSurfaceMovement::SetFFDTwist(CGeometry* geometry, CConfig* config, CFreeFormDefBox* FFDBox,
                                   CFreeFormDefBox** ResetFFDBox, unsigned short iDV, bool ResetDef) const
                                   
                                   {
  unsigned short iOrder, jOrder, kOrder;
  su2double x, y, z, movement[3], Segment_P0[3], Segment_P1[3], Plane_P0[3], Plane_Normal[3], Variable_P0, Variable_P1,
      Intersection[3], Variable_Interp;
  unsigned short index[3], iPlane, iFFDBox;
  string design_FFDBox;
  su2double Scale = config->GetOpt_RelaxFactor();

  if (rank == MASTER_NODE)
  {
    std::cout << "Current iDV: " << iDV << std::endl;
  }

  /*--- Set control points to its original value (even if the
   design variable is not in this box) ---*/

  if (ResetDef) 
  {
    /*--- All FFD boxes are reset to their original control points---*/
    for (iFFDBox = 0; iFFDBox < nFFDBox; iFFDBox++) ResetFFDBox[iFFDBox]->SetOriginalControlPoints();
  }

  design_FFDBox = config->GetFFDTag(iDV);

  // NOTE: N_out is less than the total number of FFD planes since bounding planes are disregarded.

  if (rank == MASTER_NODE)
  {
    std::cout <<"About to print chord arary " <<std::endl;
    std::cout <<"Number of slices: " << Num_slice << std::endl;
    for (int i = 0; i < Num_slice; ++i)
    {
          std::cout << "Slice index:" << chord_array_[i][0]
                        << ", Y = " << chord_array_[i][1]
                        << ", Xmin = " << chord_array_[i][2]
                        << ", Xmax = " << chord_array_[i][3]
                        << ", Chord Length = " << chord_array_[i][4]
                        << ", Quarter chord = " << chord_array_[i][5]
                        << ", Z location = " << chord_array_[i][6]
                        << std::endl;
    }
  }

  /*--- Check if the design variable applies to this FFD box---*/
  if (design_FFDBox.compare(FFDBox->GetTag()) == 0) 
  {
    /*--- Check that it is possible to move the control point ---*/

    jOrder = SU2_TYPE::Int(config->GetParamDV(iDV, 1));
    for (iPlane = 0; iPlane < FFDBox->Get_nFix_JPlane(); iPlane++) 
    {
      if (jOrder == FFDBox->Get_Fix_JPlane(iPlane)) return false;
    }

  

    /*--- Line plane intersection to find the origin of rotation ---*/

    /*--- P0: First point on the line---*/
    Segment_P0[0] = config->GetParamDV(iDV, 2);
    Segment_P0[1] = config->GetParamDV(iDV, 3);
    Segment_P0[2] = config->GetParamDV(iDV, 4);

    /*--- P1: Last point on the line---*/
    Segment_P1[0] = config->GetParamDV(iDV, 5);
    Segment_P1[1] = config->GetParamDV(iDV, 6);
    Segment_P1[2] = config->GetParamDV(iDV, 7);

    iOrder = 0;
    jOrder = SU2_TYPE::Int(config->GetParamDV(iDV, 1));
    kOrder = 0;
    su2double* coord = FFDBox->GetCoordControlPoints(iOrder, jOrder, kOrder);

    /*--- Get an arbitrary point on the plane ---*/
    Plane_P0[0] = coord[0];
    Plane_P0[1] = coord[1];
    Plane_P0[2] = coord[2];

    /*---Reference plane is spanwise normal [0,1,0]---*/
    Plane_Normal[0] = 0.0;
    Plane_Normal[1] = 1.0;
    Plane_Normal[2] = 0.0;

    Variable_P0 = 0.0;
    Variable_P1 = 0.0;

    Intersection[0] = 0.0;
    Intersection[1] = 0.0;
    Intersection[2] = 0.0;

    /*--- COmpute the intersection between the line segment and the plane---*/
    bool result = geometry->SegmentIntersectsPlane(Segment_P0, Segment_P1, Variable_P0, Variable_P1, Plane_P0,
                                                   Plane_Normal, Intersection, Variable_Interp);

    /*--- result is true if scalar t is greater than 0 ---*/
    if (result) 
    {
      /*--- xyz-coordinates of a point on the line of rotation. ---*/

      su2double a = Intersection[0];
      su2double b = Intersection[1];
      su2double c = Intersection[2];

      /*--- xyz-coordinate of the line's direction vector. ---*/
      /*--- This is set by default as normal to the spanwise plane -> [0, 1, 0] ---*/

      su2double u = Plane_Normal[0];
      su2double v = Plane_Normal[1];
      su2double w = Plane_Normal[2];

      /*--- The angle of rotation is computed based on a characteristic length of the wing,
       otherwise it is difficult to compare with other length based design variables. ---*/

      su2double RefLength = config->GetRefLength();
      su2double theta = atan(config->GetDV_Value(iDV) * Scale / RefLength);

      /*--- An intermediate value used in computations. ---*/

      su2double u2 = u * u;
      su2double v2 = v * v;
      su2double w2 = w * w;
      su2double l2 = u2 + v2 + w2;
      su2double l = sqrt(l2);
      su2double cosT;
      su2double sinT;

      /*--- Change the value of the control point if move is true ---*/

      jOrder = SU2_TYPE::Int(config->GetParamDV(iDV, 1));
      for (iOrder = 0; iOrder < FFDBox->GetlOrder(); iOrder++)
        for (kOrder = 0; kOrder < FFDBox->GetnOrder(); kOrder++) {
          index[0] = iOrder;
          index[1] = jOrder;
          index[2] = kOrder;
          su2double* coord = FFDBox->GetCoordControlPoints(iOrder, jOrder, kOrder);
          x = coord[0];
          y = coord[1];
          z = coord[2];

          cosT = cos(theta);
          sinT = sin(theta);

          /*--- Apply Rodrigues' formula to compute the rotation---*/

          /*  (x,y,z) -> original FFD control point
              (a,b,c) -> intersection point of line and normal plane -> origin of rotation
              (u,v,w) -> rotation axis [0,1,0]
              cosT, sinT -> rotation magnitude
           */

          movement[0] = a * (v2 + w2) + u * (-b * v - c * w + u * x + v * y + w * z) +
                        (-a * (v2 + w2) + u * (b * v + c * w - v * y - w * z) + (v2 + w2) * x) * cosT +
                        l * (-c * v + b * w - w * y + v * z) * sinT;
          movement[0] = movement[0] / l2 - x;

          movement[1] = b * (u2 + w2) + v * (-a * u - c * w + u * x + v * y + w * z) +
                        (-b * (u2 + w2) + v * (a * u + c * w - u * x - w * z) + (u2 + w2) * y) * cosT +
                        l * (c * u - a * w + w * x - u * z) * sinT;
          movement[1] = movement[1] / l2 - y;

          movement[2] = c * (u2 + v2) + w * (-a * u - b * v + u * x + v * y + w * z) +
                        (-c * (u2 + v2) + w * (a * u + b * v - u * x - v * y) + (u2 + v2) * z) * cosT +
                        l * (-b * u + a * v - v * x + u * y) * sinT;
          movement[2] = movement[2] / l2 - z;

          /*--- Check that it is possible to move the control point ---*/

          for (iPlane = 0; iPlane < FFDBox->Get_nFix_IPlane(); iPlane++) {
            if (iOrder == FFDBox->Get_Fix_IPlane(iPlane)) {
              movement[0] = 0.0;
              movement[1] = 0.0;
              movement[2] = 0.0;
            }
          }

          for (iPlane = 0; iPlane < FFDBox->Get_nFix_KPlane(); iPlane++) {
            if (kOrder == FFDBox->Get_Fix_KPlane(iPlane)) {
              movement[0] = 0.0;
              movement[1] = 0.0;
              movement[2] = 0.0;
            }
          }

          FFDBox->SetControlPoints(index, movement);
        }
    }

    /* Print out the twist planes where no rotation was found! */
    if (!result)
    {
      if (rank == MASTER_NODE)
      {
        std::cout << "*****************************" << std::endl;
        std::cout << " SEGMENT DOES NOT INTERSECT FFD PLANE: NO ROTTION POINT FOUND!" << std::endl;
        std::cout << " Control point location: " << Plane_P0[0] <<" , "<< Plane_P0[1] <<" , "<<Plane_P0[2] << std::endl;
        std::cout << " Segment start location: " << Segment_P0[0] <<" , "<< Segment_P0[1] <<" , "<< Segment_P0[2] << std::endl;
        std::cout << " Segment end location: " << Segment_P1[0] <<" , "<< Segment_P1[1]<<" , "<< Segment_P1[2] << std::endl;
        std::cout << "*****************************" << std::endl;
      }
    }
  }
   else 
   {
   
    return false;
  }

  return true;
}