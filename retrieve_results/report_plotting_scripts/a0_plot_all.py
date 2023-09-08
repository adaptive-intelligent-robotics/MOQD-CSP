from retrieve_results.report_plotting_scripts.a_force_threshold import force_threshold_exp
from retrieve_results.report_plotting_scripts.a_rattle import no_relax
from retrieve_results.report_plotting_scripts.b_local_optimisation import relaxation_exp
from retrieve_results.report_plotting_scripts.d_force_mutation_only import force_mut_only
from retrieve_results.report_plotting_scripts.e_benchmark_params import benchmark_params
from retrieve_results.report_plotting_scripts.f_benchmark import benchmark
from retrieve_results.report_plotting_scripts.h_dqd import dqd
from retrieve_results.report_plotting_scripts.j_omg_vs_force_only import force_vs_dqd

if __name__ == '__main__':
    no_relax(plot_individually=False, exp_1=True)
    force_threshold_exp(plot_individually=False, force_thesh=True, relax_steps=True,
                        relax_steps_no_10=True)
    relaxation_exp(plot_individually=False, relax_archive_every_5=True, relax_steps=True,
                   archive_relax_no_intermediate_relax=True)

    force_mut_only(plot_individually=False, gaussian=False, lr_step_only=False,
                   lr_step_only_simple=False, compare=True)

    benchmark_params(
        plot_individually=False,
        niches_fill=True,
        n_niches=True,
        structure_initialise=True,
        batch_size=True,
    )

    benchmark(plot_individually=True, all_comp=False, compute_symmtery_stats=False)
    dqd(plot_individually=False, lr=True, relax_steps=True, batch_size=True, dqd_rattle=True)

    force_vs_dqd(plot_individually=False, exp_1=True)
