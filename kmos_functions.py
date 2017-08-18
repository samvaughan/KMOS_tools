import glob
import logging
import subprocess
from subprocess import Popen, list2cmdline
import shutil
import readline
import os 


readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")


def multiple_log(message):
    print message
    logging.info(message)


def do_catg_from_file(file_entry):
    """Get the DO_CATG for a given file"""

    if file_entry['tpl_id'] == "KMOS_spec_cal_dark" :
        return "DARK"
    elif file_entry['tpl_id'] == "KMOS_spec_cal_calunitflat" :
        if file_entry['dpr_type'] == "FLAT,OFF" : 
            return "FLAT_OFF"
        if file_entry['dpr_type'] == "FLAT,LAMP" : 
            return "FLAT_ON"
    elif file_entry['tpl_id'] == "KMOS_spec_cal_wave" :
        if file_entry['dpr_type'] == "WAVE,OFF" : 
            return "ARC_OFF"
        if file_entry['dpr_type'] == "WAVE,LAMP" : 
            return "ARC_ON"
    elif file_entry['tpl_id'] == "KMOS_spec_cal_stdstar":
        return "STD"
    elif file_entry['tpl_id'] == "KMOS_spec_obs_stare":
        return "SCIENCE"
    else:
        raise NameError("Unknown DO_CATG")
        return


def tpl_id_to_recipe_name( tpl_id ) :
    "Get the recipe name from the tpl.id"
    if tpl_id == "KMOS_spec_cal_dark" :
        return "kmos_dark"
    if tpl_id == "KMOS_spec_cal_calunitflat" :
        return "kmos_flat"
    if tpl_id == "KMOS_spec_cal_wave" :
        return "kmos_wave_cal"
    return "Unknown_TPL_ID"


def exec_commands_seq(cmds):
    "Exec commands sequentially"
    for cmd in cmds:
        multiple_log("\tRun in {0} : {1} ...".format(cmd['dir'], list2cmdline(cmd['cmd'])))
        cwd = os.getcwd()
        os.chdir(cmd['dir'])
        #subprocess.check_call(cmd['cmd'], stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
        subprocess.check_call(cmd['cmd'])
        os.chdir(cwd)

def exec_commands_par(cmds):
    "Exec commands in parallel in multiple process"
    def done(p):
        return p.poll() is not None
    def success(p):
        return p.returncode == 0
    def fail():
        sys.exit(1)

    max_task = os.sysconf('SC_NPROCESSORS_ONLN')
    processes = []
    while True:
        while cmds and len(processes) < max_task:
            cmd = cmds.pop()
            multiple_log("\tRun in {0} : {1} ...".format(cmd['dir'], list2cmdline(cmd['cmd'])))
            processes.append(Popen(cmd['cmd'],cwd=cmd['dir'],stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb')))

        for p in processes:
            if done(p):
                if success(p):
                    processes.remove(p)
                else:
                    fail()

        if not processes and not cmds:
            break
        else:
            time.sleep(0.05)

def exec_commands(cmds, par):
    "Exec commands"
    # Empty list
    if not cmds: 
        return 
    if par:
        exec_commands_par(cmds)
    else:
        exec_commands_seq(cmds)




def SAM_create_sof(filelist, recipe_name, calibration_location=None, reduced_dark_folder=None, reduced_flat_folder=None, kmos_static_calib_directory='/Data/KCLASH/Data/Static_Cals/cal', location_for_sof=None, telluric_directory=None):
    "Creates a SOF file. This is saved in location_for_sof, or else the same location as the rest of the calibration files is location_for_sof is None"
    
    ###############
    # Verifications
    if len(filelist) < 1 :
        raise NameError("Empty List")
        return
    # Same band ?
    if len(set(map(lambda x: x['band'], filelist))) != 1 :
        raise NameError("Different bands in this sof")
        return
    # Same tpl_id ?
    if len(set(map(lambda x: x['tpl_id'], filelist))) != 1 :
        raise NameError("Different TPL_IDs in this sof")
        return

    #Commented this our- SPV 15/08/17
    # # Same tpl_start ?
    # if len(set(map(lambda x: x['tpl_start'], filelist))) != 1 :
    #     raise NameError("Different TPL_STARTs in this sof")
    #     return
    ###############

    # SOF file name
    sof_basename = "{}_{}.sof".format(recipe_name, filelist[0]['tpl_start'])
    #Unless told otherwise, make the .sof file in the same location as the rest of rthe calibration files
    if location_for_sof is None:
        sof_name = "{}/{}".format(calibration_location, sof_basename)
    else:
        sof_name = "{}/{}".format(location_for_sof, sof_basename)

    # Open the file
    sof = open(sof_name, "wb")


    # Write RAW Files
    for file_entry in filelist:
        sof.write("{}\t{}\n".format(file_entry['name'], do_catg_from_file(file_entry)))

    band_lc = filelist[0]['band'].lower()
    band_3uc = filelist[0]['band']*3


    assert recipe_name in ['kmos_dark', 'kmos_flat', 'kmos_wave_cal', 'kmos_illumination', 'kmos_std_star', 'kmos_sci_red', 'kmos_combine'], 'Recipe name not understood!'
    # Add Calibrations
    if recipe_name == 'kmos_flat' :
        if reduced_dark_folder is None:
            fname='{}/BADPIXEL_DARK.fits'.format(calibration_location)
            check_file_exists(fname)
            sof.write('{} BADPIXEL_DARK\n'.format(fname))
        else:
            fname='{}/badpixel_dark.fits'.format(reduced_dark_folder)
            check_file_exists(fname)
            sof.write('{} BADPIXEL_DARK\n'.format(fname))

    elif recipe_name == 'kmos_wave_cal':
        sof.write("{}/kmos_wave_ref_table.fits  REF_LINES\n".format(kmos_static_calib_directory))
        sof.write("{}/kmos_wave_band.fits       WAVE_BAND\n".format(kmos_static_calib_directory))
        sof.write("{}/kmos_ar_ne_list_{}.fits    ARC_LIST\n".format(kmos_static_calib_directory, band_lc))

        if reduced_flat_folder is None:
            sof.write("{}/FLAT_EDGE_{}.fits   FLAT_EDGE\n".format(calibration_location, band_3uc))
            sof.write("{}/XCAL_{}.fits        XCAL\n".format(calibration_location, band_3uc))
            sof.write("{}/YCAL_{}.fits        YCAL\n".format(calibration_location, band_3uc))
        else:
            sof.write("{}/FLAT_EDGE_{}.fits   FLAT_EDGE\n".format(reduced_flat_folder))
            sof.write("{}/XCAL_{}.fits        XCAL\n".format(reduced_flat_folder))
            sof.write("{}/YCAL_{}.fits        YCAL\n".format(reduced_flat_folder))

    elif recipe_name == 'kmos_illumination':
        #Write the sof for the illumination correction. Requires XCAL, YCAL, LCAL, FLAT_EDGE and the static WAVE_BAND

        sof.write("{}/kmos_wave_band.fits       WAVE_BAND\n".format(kmos_static_calib_directory))
        sof.write("{}/LCAL_{}.fits        LCAL\n".format(calibration_location, band_3uc))
        if reduced_flat_folder is None:
            sof.write("{}/FLAT_EDGE_{}.fits   FLAT_EDGE\n".format(calibration_location, band_3uc))
            sof.write("{}/XCAL_{}.fits        XCAL\n".format(calibration_location, band_3uc))
            sof.write("{}/YCAL_{}.fits        YCAL\n".format(calibration_location, band_3uc))
        else:
            sof.write("{}/FLAT_EDGE_{}.fits   FLAT_EDGE\n".format(reduced_flat_folder))
            sof.write("{}/XCAL_{}.fits        XCAL\n".format(reduced_flat_folder))
            sof.write("{}/YCAL_{}.fits        YCAL\n".format(reduced_flat_folder))

    elif recipe_name == 'kmos_std_star':
        #Write the sof for the illumination correction. Requires XCAL, YCAL, LCAL, FLAT_EDGE and the static WAVE_BAND

        sof.write("{}/kmos_wave_band.fits       WAVE_BAND\n".format(kmos_static_calib_directory))
        sof.write("{}/kmos_spec_type.fits       SPEC_TYPE_LOOKUP\n".format(kmos_static_calib_directory))
        sof.write("{}/kmos_atmos_iz.fits       ATMOS_MODEL\n".format(kmos_static_calib_directory))
        
        
        sof.write("{}/LCAL_{}.fits        LCAL\n".format(calibration_location, band_3uc))
        sof.write("{}/ILLUM_CORR_{}.fits  ILLUM_CORR\n".format(calibration_location, band_3uc))
        sof.write("{}/MASTER_FLAT_{}.fits MASTER_FLAT\n".format(calibration_location, band_3uc))
        sof.write("{}/XCAL_{}.fits        XCAL\n".format(calibration_location, band_3uc))
        sof.write("{}/YCAL_{}.fits        YCAL\n".format(calibration_location, band_3uc))

    elif recipe_name == 'kmos_sci_red':
        #Write the sof for the illumination correction. Requires XCAL, YCAL, LCAL, FLAT_EDGE and the static WAVE_BAND
        assert telluric_directory is not None, 'Must have a telluric file location for the science reduction!'
        sof.write("{}/LCAL_{}.fits        LCAL\n".format(calibration_location, band_3uc))
        sof.write("{}/ILLUM_CORR_{}.fits  ILLUM_CORR\n".format(calibration_location, band_3uc))
        sof.write("{}/MASTER_FLAT_{}.fits MASTER_FLAT\n".format(calibration_location, band_3uc))
        sof.write("{}/XCAL_{}.fits        XCAL\n".format(calibration_location, band_3uc))
        sof.write("{}/YCAL_{}.fits        YCAL\n".format(calibration_location, band_3uc))
        sof.write("{}/kmos_wave_band.fits       WAVE_BAND\n".format(kmos_static_calib_directory))
        sof.write("{}/kmos_oh_spec_iz.fits       OH_SPEC\n".format(kmos_static_calib_directory))
        sof.write("{}/TELLURIC_IZIZIZ.fits       TELLURIC\n".format(telluric_directory))

    elif recipe_name == 'kmos_combine':
        #We don't need any other files for kmos_combine
        pass




    else:
        raise NameError('Kmos recipe "{}" not understood!'.format(recipe_name))

    # Close sof file
    sof.close()
    return sof_basename


def reduce_darks(calibration_location, dark_files, options):
    """
    Reduce a set of darks. 
    Arguments:
        my_dir: directory the .sof file will be created in
        dark files: A list of dark filenames
        options: Command line options
    """


    # Generate exection command 
    cmds=[]   
    recipe_name='kmos_dark'
    sof_name = SAM_create_sof(dark_files, recipe_name, calibration_location=calibration_location)
    log_file = "esorex_"+recipe_name+".log"
    base_cmd = ['esorex', '--suppress-prefix=TRUE', '--log-file='+log_file, '--log-dir=.', recipe_name, sof_name]
    cmds.append({'cmd': base_cmd, 'dir': my_dir})
    exec_commands(cmds, options.parallel)

def reduce_calibs(calibration_location, filelist,  recipe_name, options, reduced_dark_folder=None, reduced_flat_folder=None):
    """
    Reduce a set of calibrations. 
    Arguments:
        my_dir: directory the .sof file will be created in
        selected_sets: A list of lists. Each list contains a set of calibrations to reduce
        calib_descritpion: The type of calibration to reduce. Must be one of .
        recipe name: The name of the esorex recipe to call. 
        options: Command line options
        reduced_dark_folder. Default is None. If we've already reduced darks, which folder are they in?
        reduced_flat_folder. Default is None. If we've already reduced the flats, which folder are they in?
    """

    # Generate exection command
    cmds=[]
    sof_name = SAM_create_sof(filelist, recipe_name, calibration_location=calibration_location, reduced_dark_folder=reduced_dark_folder, reduced_flat_folder=reduced_flat_folder)
    

    log_file = "esorex_"+recipe_name+".log"
    base_cmd = ['esorex', '--suppress-prefix=TRUE', '--log-file='+log_file, '--log-dir=.', recipe_name, sof_name]
    cmds.append({'cmd': base_cmd, 'dir': my_dir})
    
    exec_commands(cmds, options.parallel)


def reduce_std_star(destination_directory, calibration_directory, filelist,  recipe_name, options, reduced_dark_folder=None, reduced_flat_folder=None, kmos_static_calib_directory='/Data/KCLASH/Data/Static_Cals/cal/'):
    """
    Reduce a standard star observation. 
    Arguments:
        my_dir: directory the .sof file will be created in
        selected_sets: A list of lists. Each list contains a set of calibrations to reduce
        calib_descritpion: The type of calibration to reduce. Must be one of .
        recipe name: The name of the esorex recipe to call. 
        options: Command line options
        reduced_dark_folder. Default is None. If we've already reduced darks, which folder are they in?
        reduced_flat_folder. Default is None. If we've already reduced the flats, which folder are they in?
    """

    #Check the destination directory exists, and make it if not
    if not os.path.exists(os.path.abspath(destination_directory)):
        os.makedirs(os.path.abspath(destination_directory))

    # Generate exection command
    cmds=[]
    sof_name = SAM_create_sof(filelist, recipe_name, calibration_directory=calibration_directory,  location_for_sof=destination_directory)
    
    log_file = "esorex_"+recipe_name+".log"
    base_cmd = ['esorex', '--suppress-prefix=TRUE', '--log-file='+log_file, '--log-dir=.', recipe_name, sof_name]
    cmds.append({'cmd': base_cmd, 'dir': destination_directory})
    
    exec_commands(cmds, options.parallel)

def reduce_science(destination_directory, calibration_directory, telluric_directory, filelist, recipe_name, options, esorex_flags=None, log_file_name=None, reduced_dark_folder=None, reduced_flat_folder=None, kmos_static_calib_directory='/Data/KCLASH/Data/Static_Cals/cal/'):
    """
    WRITE DOCS
    """

    #Check the destination directory exists, and make it if not
    if not os.path.exists(os.path.abspath(destination_directory)):
        os.makedirs(os.path.abspath(destination_directory))

    # Generate exection command
    cmds=[]
    sof_name = SAM_create_sof(filelist, recipe_name, calibration_location=calibration_directory, location_for_sof=destination_directory, telluric_directory=telluric_directory)
    
    if log_file_name is None:
        log_file = "esorex_"+recipe_name+".log"
    else:
        log_file=log_file_name

    base_cmd = ['esorex', '--suppress-prefix=TRUE', '--log-file='+log_file, '--log-dir=.', recipe_name, sof_name]

    if esorex_flags is not None:
        for flag in esorex_flags:
            base_cmd.insert(-1, flag)

    cmds.append({'cmd': base_cmd, 'dir': destination_directory})
    
    exec_commands(cmds, options.parallel)

def call_kmos_combine(destination_directory, filelist,  options, esorex_flags=None, log_file_name=None):
    """
    WRITE DOCS
    """

    # Generate exection command
    cmds=[]
    recipe_name='kmos_combine'
    sof_name = SAM_create_sof(filelist, recipe_name, location_for_sof=destination_directory)

    if log_file_name is None:
        log_file = "esorex_"+recipe_name+".log"
    else:
        log_file=log_file_name


    log_file = "esorex_"+recipe_name+".log"
    base_cmd = ['esorex', '--suppress-prefix=TRUE', '--log-file='+log_file, '--log-dir=.', recipe_name, sof_name]

    if esorex_flags is not None:
        for flag in esorex_flags:
            base_cmd.insert(-1, flag)
    


    cmds.append({'cmd': base_cmd, 'dir': destination_directory})
    
    exec_commands(cmds, options.parallel)


def check_file_exists(fname):
    #Check that these files exist:
    if not os.path.isfile(fname):
        raise NameError("{} doesn't seem to exist!".format(fname))
    else:
        return 0