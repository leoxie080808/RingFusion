# RingFusion T3 Run Sheet

Sep 27, 2026 · @Leo

## Rig values

Every fixed number T3 needs is known or set, so T3 can run now. Pitch and yaw stay at 0 until the two axis markers give their real values.

Frame: origin O is the robot center, marked on the floor. x points forward, y left, z up, the same as ROS `base_link` (REP 103). The names are only labels: the forward axis carries the 57 mm and the sideways axis carries 0.

| Value | Symbol | Number | Source |
| --- | --- | --- | --- |
| Front of the lens, forward of O | x\_F | 57 mm | CAD |
| Optical center behind the lens front | p | 0 mm | Taken at the lens front. The dToF tip is flush with it, and depth zero comes from the dToF. The calibration puts C 1.1 mm back |
| Optical center, forward of O | x\_c | 57 mm | x\_F − p |
| Optical center, sideways of O | y\_c | 0 mm | Camera on the centerline |
| Optical center, height above the floor | z\_c | 161.9 mm (6.375 in) | Measured. The old floor fit gave 159.7 mm |
| Pitch, positive = tilted down | θ | 0° for now | Axis markers, section 5 |
| Yaw, positive = turned left | ψ | 0° for now | Axis markers, section 5 |
| dToF origin, forward of O | x\_d | 58.1 mm | x\_c + 1.13 (level camera) |
| dToF origin, sideways of O | y\_d | 0 mm | y\_c |
| dToF origin, height above the floor | z\_d | 141.7 mm | z\_c − 20.2 (level camera) |

The dToF offsets come from `calibration.yaml`, translation \[0.0, 20.195, 1.13\] mm in camera axes (x right, y down, z forward). Roll does not change depth and is not needed.

## Floor layout

Mark O and one straight forward-axis line before placing any marker. Every sideways leg is squared off that line.

1. Hang a plumb bob from the robot center and mark O on the floor.
2. Hang it from a second point on the robot's centerline, as far from O as the chassis allows, and mark that too. The line through both marks is the forward axis.
3. Extend the line to at least 5.5 m ahead with a chalk line or floor tape. Every x\_m is measured along it.
4. Square each sideways leg off the axis line with a large square, or a 3-4-5 triangle (0.9, 1.2 and 1.5 m sides, each within 3 mm).
5. Don't move the robot until the session is over. If it moves, re-plumb O and the axis before going on.

The dToF cone, as the paper states it (73.5° × 60.5°, from the dToF origin 58 mm ahead of O and 142 mm up), reaches 0.75 m to each side for every metre ahead. Its top edge rises 0.58 m per metre, and its bottom edge meets the floor about 0.3 m ahead. These edges are for planning only: the live dToF footprint decides each point's cone status.

## Marker plan

Place 22 markers: the 2 axis markers, 9 more inside the cone (2 of them beyond 4.2 m), 3 on its edge, 4 outside it and 4 on depth jumps. The positions are targets. Put each marker near its target, then tape and log where it actually ended up.

&#91;embedded content: marker plan · top view to scale, 22 points\]

The shaded wedge is the cone's sideways extent. E3 and O3 sit high, at and above the cone's top edge, so check their heights in the table.

| ID | Type | x\_m forward (mm) | y\_m, left + (mm) | z\_m height (mm) | Placement |
| --- | --- | --- | --- | --- | --- |
| A1 | Axis marker | 2000 | 0 | 162 | On the axis line, center at lens height |
| A2 | Axis marker | 4000 | 0 | 162 | On the axis line, center at lens height |
| C1 | In cone | 600 | 80 | 200 |  |
| C2 | In cone | 1000 | 150 | 300 |  |
| C3 | In cone | 1500 | −300 | 100 |  |
| C4 | In cone | 2000 | 400 | 500 |  |
| C5 | In cone | 2500 | −500 | 50 |  |
| C6 | In cone | 3000 | 600 | 800 |  |
| C7 | In cone | 3500 | −400 | 300 |  |
| C8 | In cone, far | 4500 | 300 | 400 | Beyond 4.2 m |
| C9 | In cone, far | 5000 | −300 | 200 | Beyond 4.2 m |
| E1 | Cone edge, left | 1500 | 1080 | 162 | Slide sideways until it sits on the footprint edge |
| E2 | Cone edge, right | 2500 | −1820 | 162 | Slide sideways until it sits on the footprint edge |
| E3 | Cone edge, top | 2200 | 200 | 1390 | Raise or lower onto the top edge |
| O1 | Outside, right | 800 | −1000 | 162 |  |
| O2 | Outside, left | 1500 | 1600 | 300 |  |
| O3 | Outside, above | 2500 | 300 | 1900 |  |
| O4 | Outside, right | 1200 | −1400 | 600 |  |
| D1 | Depth jump, thin pole | 2000 | −600 | 162 | Target on the pole, open space behind |
| D2 | Depth jump, box corner | 1500 | 500 | 300 | Target on the corner, wall behind |
| D3 | Depth jump, door frame | 3000 | −900 | 1000 | Target on the frame edge |
| D4 | Depth jump, chair back | 2500 | 800 | 700 | Target on the chair's edge |

- Use a printed cross or AprilTag about 15 cm across, so the 5 m targets stay sharp.
- Every point is held out (set = heldout), since the uncertainty constants are frozen.
- A depth jump needs the background more than 0.15 m behind the target, the edge protocol's threshold. Aim for 0.5 m or more.
- Outside points must still show in the pipeline's depth image. If one doesn't, move it inward until it does, keeping it off the footprint.
- If several markers stay up at once, check that none hides another from the camera.

## Measuring and capturing

Measure every marker first, then capture the whole scene 5 times. With all markers up at once, one capture covers every point, so the capture itself takes about 10 minutes.

**Tools:** a steel tape with 1 mm graduations and at least 5.5 m long, a plumb bob, a large square or a 3-4-5 triangle, a spirit level, a chalk line or floor tape, and stands tall enough for O3 (1.9 m).

**Each point**

1. Put the marker near its target from the plan.
2. Tape three legs from O, each read to 1 mm: x\_m along the axis line, y\_m squared off the line to the point under the marker center (left positive), and z\_m from the floor to the marker center (with the level or the plumb bob).
3. Check its cone status on the live dToF footprint and note it as in, edge or out. The plan's category is only the intent.

**Capture**

1. Warm up the Jetson and the dToF for 10 minutes, and keep the lighting constant.
2. Take 5 captures, at least 2 minutes apart. Don't touch the robot or any marker between measuring and capturing.
3. Save the raw frames and dToF packets, and log one row per point per capture. Lengths are in mm, and u\_px and v\_px are on the rectified image. Leave Z\_ref empty in the field; the script fills it from the legs and the rig values.

```csv
point_id,session,capture,set,cone_status,x_m,y_m,z_m,x_c,y_c,z_c,pitch_deg,yaw_deg,u_px,v_px,Z_ref,depth_pred,sigma_pred,dtof_raw,V_fit,tau2,term_arb,term_angle,term_nu,floor_applied,omega,alpha_deg,anchor_count,cond_number,frame_dropped
A1,T3-1,1,heldout,in,2000,0,162,57,0,161.9,0,0,,,,,,,,,,,,,,,,,
```

The example row shows where the rig values go. dtof\_raw is the raw reading of the zone covering the marker, and alpha\_deg is the angle to the nearest anchor.

## Turning legs into depth

The script turns each point's legs into Z, its depth along the camera's optical axis, which is what the pipeline outputs. With pitch and yaw at 0 this is simply Z = d.

```latex
d = x_m - x_c, \qquad y = y_m - y_c, \qquad h = z_c - z_m
```

```latex
Z = \cos\theta \, (d \cos\psi + y \sin\psi) + h \sin\theta
```

Pitch and yaw come from the axis markers A1 and A2, using each marker's pixel (u, v) on the rectified image and the rectified camera matrix (f\_x, f\_y, c\_x, c\_y):

```latex
\theta = -\arctan\frac{v - c_y}{f_y}, \qquad \psi = \arctan\left(\frac{u - c_x}{f_x}\cos\theta\right)
```

A1 and A2 should agree within about 0.2°. Put their mean in the rig card and re-run the script. The slant range r = √(d² + y² + h²) is a cross-check only, never the reference.

| Source of reference error | 1σ (mm) |
| --- | --- |
| Tape reading, 1 mm graduations | 0.3 |
| Locating O and the forward axis | 1 |
| CAD offset to the lens front | 1 |
| Optical center inside the lens (taken at the lens front) | 2 |
| Locating the marker center | 1 |
| Combined, root sum of squares | 2.7 |

Expect about 3 to 4 mm at 4 m, since small angle errors grow with distance. The pipeline's in-cone error is about 30 mm, so the reference is roughly ten times smaller than what it measures.
