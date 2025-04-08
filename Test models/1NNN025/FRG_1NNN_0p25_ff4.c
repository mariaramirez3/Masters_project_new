#include <diverge.h>
#include <time.h>
#include <stdlib.h>  // For atof()

//reads NNN TBM file instead
diverge_model_t* MODEL( index_t nk, index_t nkf , double U , double mu); //This is defined under the main function.

int main( int argc, char* argv[] ) {

    // User specification, can modify for each system depending on what parameters you're interested in. 
    if (argc != 5)
    {mpi_printf("You've forgotten input parameters... use  ./testing.c U J nk nkf");
    exit(0);
    }
    double U = atof(argv[1]);
    double mu = atof(argv[2]);  // maybe change to mu 
    double nk = atof(argv[3]);
    double nkf = atof(argv[4]);
    

    char outputname[80];
    char modelname[80];
    // Format the string
    sprintf(outputname, "1NNN_%d_%d_%.2f_%.2f_out.dvg", (int)nk,(int)nkf,U,mu);
    sprintf(modelname, "1NNN_%d_%d_%.2f_%.2f_mod.dvg", (int)nk,(int)nkf,U,mu);



    //This Initialization bit should never be changed. 
    diverge_init(&argc,&argv);
    diverge_compilation_status();
    

    //This is where we create the model object with all the details for calculations (Defined below main function).
    diverge_model_t* m = MODEL(nk,nkf,U,mu);

    //This needs to be big enough that the results don't change. 
    double tu_form_factor_cutoff = 4.0;


    diverge_model_internals_common( m );
    diverge_model_internals_tu(m, tu_form_factor_cutoff);

    // common internals, chempot
    diverge_model_set_chempot(m, NULL, NULL, mu);
    // output
    
    // post-process model output This saves a mod.dvg file that we can plot to make sure the model works well. Also saves all the additional information about the model that DIVERGE needs.
    diverge_model_output_conf_t model_defaults = diverge_model_output_conf_defaults_CPP();
    model_defaults.kc_ibz_path=1;
    model_defaults.kf_ibz_path=1;
    model_defaults.kc=1;
    model_defaults.kf=1;
    //mpi_usr_printf( "model@%s\n", diverge_model_to_file_finegrained(m, modelname,&model_defaults) );


    // flow step & integrate. THE MOST IMPORTANT PART, where all the FRG occurs. 
    diverge_flow_step_t* f = diverge_flow_step_init(m, "tu", "PCD");
    diverge_euler_t eu = diverge_euler_defaults;
    eu.maxvert = 200.0; 
    eu.dLambda_fac = 0.08; //This defines the max value, 50 is default, but large U requires larger value. 
    double cmax[3] = {0}; 
    double vmax = 0.0;
    mpi_printf("Lambda,P,C,D,max\n");
    do {
        diverge_flow_step_euler( f, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( f, &vmax );
        diverge_flow_step_chanmax(f, cmax);
        mpi_printf( "%.5e %.5e %.5e %.5e %.5e\n", eu.Lambda,cmax[0],cmax[1],cmax[2], vmax );
    } while (diverge_euler_next(&eu, vmax));

    // post-processing of the flow. Saves the final results to a _out.dvg file that we can use to look at susceptibilities etc.
    diverge_postprocess_conf_t postprocess_defaults = diverge_postprocess_conf_defaults_CPP();
    postprocess_defaults.tu_susceptibilities_full = true;

    //diverge_postprocess_and_write_finegrained( f, outputname,&postprocess_defaults);

    // cleanup and done. 
    diverge_flow_step_free(f);
    diverge_model_free(m);
    diverge_finalize();
   
    return 0;
}


diverge_model_t* MODEL( index_t nk, index_t nkf , double U , double mu) 
{
    diverge_model_t* m = diverge_model_init();
    strcpy(m->name, __FILE__ );

    double a_lat = 1;
    double b_lat = 1;
    double c_lat = 1;
    
    int n_atoms = 1;
    int n_orbitals_per_atom = 1;

    
    // setting k-points, lattice constants, number of orbitals. 
    // IF you have a 2D model, delete the lines m->nk[2] and m->nkf[2]. It will speed up the calculation. 
    m->nk[0] = nk;
    m->nk[1] = nk;
    //m->nk[2] = nk;
    m->nkf[0] = nkf;
    m->nkf[1] = nkf;
    //m->nkf[2] = nkf;
   
    m->n_orb = n_atoms*n_orbitals_per_atom; //This doesn't need to be specified, as it will be overwritten when DIVERGE reads in the Wannier90 file 

    //need to specify lattice vectors (a 3x3 matrix) all non-specified elements are set to zero.
    m->lattice[0][0] = a_lat;
    m->lattice[1][1] = b_lat;
    m->lattice[2][2] = c_lat;

    //Positions can be found in .wout file from Wannier90.
    //make as many m->positions[i] as there are orbitals in the unit cell.
    //need to make this automatic eventually...
    m->positions[0][0] = 0.0;
    m->positions[0][1] = 0.0;
    m->positions[0][2] = 0.0;
    /*
    m->positions[1][0] = 1.93065;
    m->positions[1][1] = 1.93065;
    m->positions[1][2] = 0.0;

    m->positions[2][0] = 1.93065;
    m->positions[2][1] = 1.93065;
    m->positions[2][2] = 0.0;
    */


    //No SOC
    m->SU2 = 1; //true
    m->n_spin = 1;

    // SOC
    //m->SU2 = 0; //false
    //m->n_spin = 2;
    
    
    //specify a k-path for the band structure that will be saved in the model output file and the susceptibility that will be calculated in the post-processing part o the code
    //Useful to check if the band structure is being read correctly.
    m->n_ibz_path = 4;
    //Gamma
    m->ibz_path[0][0] = 0.0;
    m->ibz_path[0][1] = 0.0;
    m->ibz_path[0][2] = 0.0;
    //X
    m->ibz_path[1][0] = 0.5;
    m->ibz_path[1][1] = 0.5;
    m->ibz_path[1][2] = 0.0;
    //M
    m->ibz_path[2][0] = 0.5;
    m->ibz_path[2][1] = 0.0;
    m->ibz_path[2][2] = 0.0;
    //Gamma
    m->ibz_path[3][0] = 0.0;
    m->ibz_path[3][1] = 0.0;
    m->ibz_path[3][2] = 0.0;
    
    //Tight binding model goes here
    m->hop = diverge_read_W90_C("1NNN_0p25_hr.dat",0,&m->n_hop, &m->n_orb);
    

    //This is the important bit that defines the interaction hamiltonian! 
    //As long as you only care about Hubbard-Kanamori, this doesn't need to be touched. 

    // R here indicates that they are on thge same atom

    double J = 0;
    m->vert = diverge_mem_alloc_rs_vertex_t(1024);
    for (index_t atom1=0; atom1 < n_atoms; ++atom1)
    {
        for (index_t o1 = 0; o1 < n_orbitals_per_atom; ++o1)
        {
            for (index_t o2 = 0; o2 < n_orbitals_per_atom; ++o2)
            {
                if(o1 == o2)
                {
                
                    m->vert[m->n_vert++] = (rs_vertex_t){.chan='D',.s1=-1, .R={0,0,0}, .o1=o1+n_orbitals_per_atom*atom1, .o2=o2+n_orbitals_per_atom*atom1, .V=U};
                }
                else
                {
                    m->vert[m->n_vert++] = (rs_vertex_t){.chan='D',.s1=-1, .R={0,0,0}, .o1=o1+n_orbitals_per_atom*atom1, .o2=o2+n_orbitals_per_atom*atom1, .V=U-2*J};
                    m->vert[m->n_vert++] = (rs_vertex_t){.chan='C',.s1=-1, .R={0,0,0}, .o1=o1+n_orbitals_per_atom*atom1, .o2=o2+n_orbitals_per_atom*atom1, .V=J};
                    m->vert[m->n_vert++] = (rs_vertex_t){.chan='P',.s1=-1, .R={0,0,0}, .o1=o1+n_orbitals_per_atom*atom1, .o2=o2+n_orbitals_per_atom*atom1, .V=J};
                }
            }
        }
    }
    
        
    // generate symmetries
    //Need to specify the orbital character of each orbital. Then include symmetries of the point group of your system. (add/remove as appropriate).
    site_descr_t* sites = (site_descr_t*)calloc(3, sizeof(site_descr_t));

    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;
    /*
    sites[1].n_functions = 1;
    sites[1].amplitude[0] = 1.;
    sites[1].function[0] = orb_dyz;

    sites[2].n_functions = 1;
    sites[2].amplitude[0] = 1.;
    sites[2].function[0] = orb_dxy;
    */

    index_t symsize = POW2(m->n_orb*m->n_spin);
    m->orb_symmetries = (complex128_t*)calloc(20*symsize,sizeof(complex128_t));
    sym_op_t curr_symm;

    //Rotations around z
    curr_symm.type = 'R'; 
    curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 0.;
     curr_symm.normal_vector[2] = 1.;

    //Identity
    
    curr_symm.angle = 0;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //C4(z)
    curr_symm.angle = 90;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //C4_2(z)
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //C4_3(z)
    curr_symm.angle = 270;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //C2(x)
    curr_symm.normal_vector[0] = 1.;
    curr_symm.normal_vector[1] = 0.;
    curr_symm.normal_vector[2] = 0.;
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //C2(y)
    curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 1.;
    curr_symm.normal_vector[2] = 0.;
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //C2(xy)
    curr_symm.normal_vector[0] = 1.;
    curr_symm.normal_vector[1] = 1.;
    curr_symm.normal_vector[2] = 0.;
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //C2(x-y)
    curr_symm.normal_vector[0] = 1.;
    curr_symm.normal_vector[1] = -1.;
    curr_symm.normal_vector[2] = 0.;
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;


    //inversion
    curr_symm.type = 'I';
    curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 0.;
    curr_symm.normal_vector[2] = 1.;
    curr_symm.angle = 0;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //mirror(xy)
    curr_symm.type = 'M'; //for rotoinversion
    curr_symm.normal_vector[0] = 1.;
    curr_symm.normal_vector[1] = 1.;
    curr_symm.normal_vector[2] = 0.;
    curr_symm.angle = 0;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[m->n_sym][0][0]),
                                    m->orb_symmetries+(m->n_sym)*symsize);
    m->n_sym++;

    //didn't add in the final mirror syms. 

    //This has to go at the end! ensures the final model is resymmetrised.
    m->n_sym*= -1;

    return m;
}




