# AI Use Log — HydroSense-Kenya
## ICS 2207 Scientific Computing — Sharon

## Entry 1

 Field ->Details 

 **Date** -> 20 May 2026 
 **Task** -> Our src/ files were empty placeholders and our CSV datasets had dummy headers — we needed help getting the actual working code and data 
 **Tool** -> Claude 
 **Prompt** -> "Our numerical_methods.py only has a comment, our weather CSV only has header1,header2,header3 — we need the actual implementations and real dataset to get started" 
 **AI Output Summary** -> Generated the full src/ module files and the three datasets (weather_daily.csv, soil_sensor_data.csv, crop_zone_parameters.csv) with realistic March 2026 Kenya data 
 **Modifications** -> After getting the files working, I went through each function in numerical_methods.py to understand it before using it in the notebook. I wrote all the problem framing, interpretations, and scientific commentary myself. I also fixed a bug where np.trapz had been renamed to np.trapezoid in our version of NumPy — Claude had not accounted for that. I also added the dry scenario comparison in Level 5 myself to make the optimizer results more meaningful since the actual March data required 0mm irrigation 
 **Validation** -> Ran each numerical method against known test cases. Verified ET values were physically realistic for Kenya (2-4 mm/day). Confirmed Gaussian elimination matched np.linalg.solve. Checked Monte Carlo results manually against expected probabilities 


## Entry 2

 Field -> Details 

 **Date** -> 20 May 2026 
 **Task** -> Getting the Jupyter notebook cells working after importing from src/ kept failing with ImportError 
 **Tool** ->Claude 
 **Prompt** -> "I keep getting ImportError: cannot import name bisection from src.numerical_methods even though the file is there" 
**AI Output Summary** ->Identified that the Jupyter kernel was caching the old empty version of the file and suggested restarting the kernel to force it to reload the updated src/ files 
 **Modifications** -> Once imports worked I built all the notebook cells myself — the root finding experiment with the irrigation problem, the convergence plot, the differentiation analysis, the integration comparison and the Monte Carlo simulation were all structured and interpreted by me 
| **Validation** -> After kernel restart confirmed all imports loaded successfully. Ran every cell from top to bottom without errors to verify full reproducibility 