/* Author: Hamed Babaei, 2018 */

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/std_cxx11/shared_ptr.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/constrained_linear_operator.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <iostream>
#include <fstream>
//////////////////////////////////////////////////////////////////
namespace PhaseField
{
  using namespace dealii;

  typedef TrilinosWrappers::MPI::Vector vectorType;
  typedef TrilinosWrappers::SparseMatrix matrixType;

// INPUT OF PARAMETERS
  namespace Parameters
  {
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;
      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree", "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");
        prm.declare_entry("Quadrature order", "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }
////////////////////////////////////////////////////
    struct Geometry
    {
      unsigned int refinement;
      double       scale;
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Global refinement", "4",
                           Patterns::Integer(0),
                           "Global refinement level");
        prm.declare_entry("Grid scale", "1",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");
      }
      prm.leave_subsection();
    }
    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        refinement = prm.get_integer("Global refinement");
        scale = prm.get_double("Grid scale");
      }
      prm.leave_subsection();
    }

 /////////////////////////////////////////////////
        struct Materials
            {
              double lambdaA; // austenite phase
              double muA;     // austenite phase
              double lambdaM; // martensite phase
              double muM;     // martensite phase
              double L;        // interface mobility
              double A;       // parameter for interaction energy
              double delta_psi;  //thermal energy jump
              double ki0;  //threshold

              static void
              declare_parameters(ParameterHandler &prm);
              void
              parse_parameters(ParameterHandler &prm);
            };
            void Materials::declare_parameters(ParameterHandler &prm)
            {
              prm.enter_subsection("Material properties");
              {
                prm.declare_entry("lambda austenite", "144.0", /*  */
                                  Patterns::Double(),
                                  "lambda austenite");
                prm.declare_entry("mu austenite", "74.0",
                                  Patterns::Double(0.0),
                                  "mu austenite");
                prm.declare_entry("lambda martensite", "379.0", /*  */
                                  Patterns::Double(),
                                  "lambda martensite");
                prm.declare_entry("mu martensite", "134.0",
                                  Patterns::Double(0.0),
                                  "mu martensite");
                prm.declare_entry("kinetic coeff", "2.6",
                                  Patterns::Double(0.0),
                                  "kinetic coeff");
                prm.declare_entry("interaction parameter", "0.028",
                                  Patterns::Double(),
                                  "interaction parameter");
                prm.declare_entry("thermal jump", "0.028",
                                  Patterns::Double(),
                                  "thermal jump");
                prm.declare_entry("threshold", "0.028",
                                   Patterns::Double(),
                                   "threshold");

              }
              prm.leave_subsection();
            }

            void Materials::parse_parameters(ParameterHandler &prm)
            {
              prm.enter_subsection("Material properties");
              {
                lambdaA = prm.get_double("lambda austenite");
                muA = prm.get_double("mu austenite");
                lambdaM = prm.get_double("lambda martensite");
                muM = prm.get_double("mu martensite");
                L = prm.get_double("kinetic coeff");
                A = prm.get_double("interaction parameter");
                delta_psi = prm.get_double("thermal jump");
                ki0 = prm.get_double("threshold");
              }
              prm.leave_subsection();
            }


    /////////////////////////////////////////////////
    struct Time
    {
      double delta_t;
      double end_time;
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "50",
                          Patterns::Double(),
                          "End time");
        prm.declare_entry("Time step size", "0.01",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }
    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }
 ///////////////////////////////////////////////////////
    struct AllParameters : public FESystem,
      public Geometry,
      public Materials,
      public Time
    {
      AllParameters(const std::string &input_file);
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }
    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      Time::declare_parameters(prm);
    }
    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      Time::parse_parameters(prm);
    }
  }

 //  DEFINE SECOND ORDER IDENTITY, AND TWO FOURTH ORDER IDENTITY TENSORS
  template <int dim>
  class StandardTensors
  {
  public:
    static const SymmetricTensor<2, dim> I;
    static const SymmetricTensor<4, dim> IxI;
    static const SymmetricTensor<4, dim> II;
   //   static const SymmetricTensor<2, dim> transformation_strain;
  };
  template <int dim>
  const SymmetricTensor<2, dim>
  StandardTensors<dim>::I = unit_symmetric_tensor<dim>();
  template <int dim>
  const SymmetricTensor<4, dim>
  StandardTensors<dim>::IxI = outer_product(I, I);
  template <int dim>
  const SymmetricTensor<4, dim>
  StandardTensors<dim>::II = identity_tensor<dim>();

// DEFINE TIME STEP, CURRENT TIME ETC.
  class Time
  {
  public:
    Time (const double time_end,
          const double delta_t)
      :
      timestep(0),
      time_current(0.0),
      time_end(time_end),
      delta_t(delta_t),
      strainstep(0)

    {}

    virtual ~Time()
    {}
    double current() const
    {
      return time_current;
    }
    double end() const
    {
      return time_end;
    }
    double get_delta_t() const
    {
      return delta_t;
    }
    unsigned int get_timestep() const
    {
      return timestep;
    }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    const double delta_t;
    unsigned int strainstep;
  };

//////////// COMPUTE ELASTIC MODULUS AND STRESSES
    template <int dim>
     class Material_Constitutive
     {
     public:
       Material_Constitutive(const double lambda_A_iso,
                                                                             const double mu_A_iso,
                                                  const double lambda_M_iso,
                                                  const double mu_M_iso,
                                                  const double A,
                                                  const double delta_psi)
         :
         det_F(1.0),
         Fe(Tensor<2, dim>()),
         ge(StandardTensors<dim>::I),
         Be(SymmetricTensor<2, dim>()),
         I1(0.0),

         Ge(StandardTensors<dim>::I),
         Ee(Tensor<2, dim>()),
         FeEe(Tensor<2, dim>()),
         EeEe(Tensor<2, dim>()),

         lambda_A_iso(lambda_A_iso),
         mu_A_iso(mu_A_iso),
         lambda_M_iso(lambda_M_iso),
         mu_M_iso(mu_M_iso),

         C_A(Vector<double> (9)),
         C_M1(Vector<double> (9)),
         C_M2(Vector<double> (9)),
         C_M3(Vector<double> (9)),
         lambda_A(Vector<double> (3)),
         lambda_M1(Vector<double> (3)),
         lambda_M2(Vector<double> (3)),
         lambda_M3(Vector<double> (3)),
         lambda (Vector<double> (3)),
         mu_A(Vector<double> (3)),
         mu_M1(Vector<double> (3)),
         mu_M2(Vector<double> (3)),
         mu_M3(Vector<double> (3)),
         mu (Vector<double> (3)),
         nu_A(Vector<double> (3)),
         nu_M1(Vector<double> (3)),
         nu_M2(Vector<double> (3)),
         nu_M3(Vector<double> (3)),
         nu (Vector<double> (3)),

         A0(A),
         delta_psi0(delta_psi)


       {}

       ~Material_Constitutive()
       {}

       void update_material_data (const Tensor<2, dim> &F, const Tensor<2, dim> &F_e, const double &c_1,const double &c_2,const double &c_3 )
       {
         Fe=F_e;
         det_F = determinant(F);
         ge = symmetrize(Fe*transpose(Fe));
         Ge = symmetrize(transpose(Fe)*Fe);
         Ee = 0.5*(Ge-StandardTensors<dim>::I);
         I1 = trace(Ee);
         FeEe= Fe*Ee;
         EeEe= Ee*Ee;

         c_total=c_1+c_2+c_3;
         c0=1-c_total;
         c1=c_1;
         c2=c_2;
         c3=c_3;

// Parameters related to the calibration of instability criteria using Athermal Threshhold (Kd and kr)
         ///sr=6
//         kd1=0.1753 - 0.0208333 *A0 - 0.0208333 *delta_psi0;
//         kr1=0.149153 + 0.0208333 *A0 - 0.0208333 *delta_psi0;
//         kd3=-0.447 + 0.125 *A0+ 0.125 *delta_psi0;
//         kr3=-0.808318 - 0.166667 *A0+ 0.166667 *delta_psi0;
         //sr=-8
//         kd1= 0.1753 - 0.0208333*A0 - 0.0208333*delta_psi0;
//         kr1= 0.149153 - 0.015625*A0 + 0.015625*delta_psi0;
//         kd3= -0.447 + 0.125*A0 + 0.125*delta_psi0;
//         kr3= -0.808318 + 0.125 *A0 - 0.125*delta_psi0;

         ///sr=1
         kd1=0.1753 - 0.0208333 *A0 - 0.0208333 *delta_psi0;
         kr1=0.149153 + 0.125 *A0 - 0.125 *delta_psi0;
         kd3=-0.447 + 0.125  *A0 + 0.125 *delta_psi0;
         kr3=-0.808318 - 1.  *A0 + 1.*delta_psi0;

// Elstic constants for orthotropic material
        C_A[0]= 167.5;//C_A_11
         C_A[1]= 167.5;//C_A_22
         C_A[2]= 167.5;//C_A_33
         C_A[3]=  80.1;//C_A_44
         C_A[4]=  80.1;//C_A_55
         C_A[5]=  80.1;//C_A_66
         C_A[6]=  65.0;//C_A_12
         C_A[7]=  65.0;//C_A_13
         C_A[8]=  65.0;//C_A_23

        C_M1[0]= 174.76;//C_M1_11
         C_M1[1]= 174.76;//C_M1_22
         C_M1[2]= 136.68;//C_M1_33
         C_M1[3]=  60.24;//C_M1_44
         C_M1[4]=  60.24;//C_M1_55
         C_M1[5]=  42.22;//C_M1_66
         C_M1[6]= 102.00;//C_M1_12
         C_M1[7]=  68.00;//C_M1_13
         C_M1[8]=  68.00;//C_M1_23

         C_M2[0]= 174.76;//C_M2_11
         C_M2[1]= 136.68;//C_M2_22
         C_M2[2]= 174.76;//C_M2_33
         C_M2[3]=  60.24;//C_M2_44
         C_M2[4]=  42.22;//C_M2_55
         C_M2[5]=  60.24;//C_M2_66
         C_M2[6]=  68.00;//C_M2_12
         C_M2[7]= 102.00;//C_M2_13
         C_M2[8]=  68.00;//C_M2_23

         C_M3[0]= 136.68;//C_M3_11
         C_M3[1]= 174.76;//C_M3_22
         C_M3[2]= 174.76;//C_M3_33
         C_M3[3]=  42.22;//C_M3_44
         C_M3[4]=  60.24;//C_M3_55
         C_M3[5]=  60.24;//C_M3_66
         C_M3[6]=  68.00;//C_M3_12
         C_M3[7]=  68.00;//C_M3_13
         C_M3[8]= 102.00;//C_M3_23


         lambda_A[0]= C_A[0]+C_A[8]+2*C_A[3]-(C_A[6]+C_A[7]+2*C_A[4]+2*C_A[5]);
         lambda_A[1]= C_A[1]+C_A[7]+2*C_A[4]-(C_A[6]+C_A[8]+2*C_A[3]+2*C_A[5]);
         lambda_A[2]= C_A[2]+C_A[6]+2*C_A[5]-(C_A[7]+C_A[8]+2*C_A[3]+2*C_A[4]);

         mu_A[0]= 0.5*(C_A[6]+C_A[7]-C_A[8]);
         mu_A[1]= 0.5*(C_A[6]+C_A[8]-C_A[7]);
         mu_A[2]= 0.5*(C_A[7]+C_A[8]-C_A[6]);

         nu_A[0]= 0.5*(C_A[4]+C_A[5]-C_A[3]);
         nu_A[1]= 0.5*(C_A[3]+C_A[5]-C_A[4]);
         nu_A[2]= 0.5*(C_A[3]+C_A[4]-C_A[5]);

         lambda_M1[0]= C_M1[0]+C_M1[8]+2*C_M1[3]-(C_M1[6]+C_M1[7]+2*C_M1[4]+2*C_M1[5]);
         lambda_M1[1]= C_M1[1]+C_M1[7]+2*C_M1[4]-(C_M1[6]+C_M1[8]+2*C_M1[3]+2*C_M1[5]);
         lambda_M1[2]= C_M1[2]+C_M1[6]+2*C_M1[5]-(C_M1[7]+C_M1[8]+2*C_M1[3]+2*C_M1[4]);

         mu_M1[0]= 0.5*(C_M1[6]+C_M1[7]-C_M1[8]);
         mu_M1[1]= 0.5*(C_M1[6]+C_M1[8]-C_M1[7]);
         mu_M1[2]= 0.5*(C_M1[7]+C_M1[8]-C_M1[6]);

         nu_M1[0]= 0.5*(C_M1[4]+C_M1[5]-C_M1[3]);
         nu_M1[1]= 0.5*(C_M1[3]+C_M1[5]-C_M1[4]);
         nu_M1[2]= 0.5*(C_M1[3]+C_M1[4]-C_M1[5]);

         lambda_M2[0]= C_M2[0]+C_M2[8]+2*C_M2[3]-(C_M2[6]+C_M2[7]+2*C_M2[4]+2*C_M2[5]);
         lambda_M2[1]= C_M2[1]+C_M2[7]+2*C_M2[4]-(C_M2[6]+C_M2[8]+2*C_M2[3]+2*C_M2[5]);
         lambda_M2[2]= C_M2[2]+C_M2[6]+2*C_M2[5]-(C_M2[7]+C_M2[8]+2*C_M2[3]+2*C_M2[4]);

         mu_M2[0]= 0.5*(C_M2[6]+C_M2[7]-C_M2[8]);
         mu_M2[1]= 0.5*(C_M2[6]+C_M2[8]-C_M2[7]);
         mu_M2[2]= 0.5*(C_M2[7]+C_M2[8]-C_M2[6]);

         nu_M2[0]= 0.5*(C_M2[4]+C_M2[5]-C_M2[3]);
         nu_M2[1]= 0.5*(C_M2[3]+C_M2[5]-C_M2[4]);
         nu_M2[2]= 0.5*(C_M2[3]+C_M2[4]-C_M2[5]);

         lambda_M3[0]= C_M3[0]+C_M3[8]+2*C_M3[3]-(C_M3[6]+C_M3[7]+2*C_M3[4]+2*C_M3[5]);
         lambda_M3[1]= C_M3[1]+C_M3[7]+2*C_M3[4]-(C_M3[6]+C_M3[8]+2*C_M3[3]+2*C_M3[5]);
         lambda_M3[2]= C_M3[2]+C_M3[6]+2*C_M3[5]-(C_M3[7]+C_M3[8]+2*C_M3[3]+2*C_M3[4]);

         mu_M3[0]= 0.5*(C_M3[6]+C_M3[7]-C_M3[8]);
         mu_M3[1]= 0.5*(C_M3[6]+C_M3[8]-C_M3[7]);
         mu_M3[2]= 0.5*(C_M3[7]+C_M3[8]-C_M3[6]);

         nu_M3[0]= 0.5*(C_M3[4]+C_M3[5]-C_M3[3]);
         nu_M3[1]= 0.5*(C_M3[3]+C_M3[5]-C_M3[4]);
         nu_M3[2]= 0.5*(C_M3[3]+C_M3[4]-C_M3[5]);

         for (unsigned int n=0; n<3; ++n)
         {
         lambda[n] = lambda_A[n]*c0+lambda_M1[n]*c1+lambda_M2[n]*c2+lambda_M3[n]*c3;
         mu[n] = mu_A[n]*c0+mu_M1[n]*c1+mu_M2[n]*c2+mu_M3[n]*c3;
         nu[n] = nu_A[n]*c0+nu_M1[n]*c1+nu_M2[n]*c2+nu_M3[n]*c3;
         }

// In the case of isotropic material, elstic constants are computed as follows
         lambda_iso = lambda_A_iso+ lambda_M_iso*c_total;
        mu_iso = mu_A_iso+mu_M_iso*c_total;


         Assert(det_F > 0, ExcInternalError());
       }

// Compute the Kirchhoff stress and Jacobians for Orthotropic material
           SymmetricTensor<2, dim> get_tau() const
           {
            SymmetricTensor<2, dim> kirchhoff_stress;
            for (unsigned int n=0; n<dim; ++n)
              for (unsigned int i=0; i<dim; ++i)
                for (unsigned int j=0; j<dim; ++j)

                    kirchhoff_stress[i][j] += lambda[n]*Ee[n][n]*Fe[i][n]*Fe[j][n]+
                                              mu[n]*(I1*Fe[i][n]*Fe[j][n]+Ee[n][n]*ge[i][j])+
                                              2*nu[n]*(Fe[i][n]*FeEe[j][n]+FeEe[i][n]*Fe[j][n]);

           return kirchhoff_stress;
           }
//compute the total Jc
           SymmetricTensor<4, dim> get_Jc() const
           {

             SymmetricTensor<4,dim> elasticityTensor;

                for (unsigned int n=0; n<dim; ++n)
                  for (unsigned int i=0; i<dim; ++i)
                    for (unsigned int j=0; j<dim; ++j)
                      for (unsigned int k=0; k<dim; ++k)
                        for (unsigned int l=0; l<dim; ++l)
                        {
                          elasticityTensor[i][j][k][l] +=
                             lambda[n]*Fe[i][n]*Fe[j][n]*Fe[k][n]*Fe[l][n]+
                                 mu[n]*(Fe[i][n]*Fe[j][n]*ge[k][l]+ ge[i][j]*Fe[k][n]*Fe[l][n])+
                                 nu[n]*(Fe[i][n]*ge[j][k]*Fe[l][n]+ Fe[j][n]*ge[i][k]*Fe[l][n]+
                                        Fe[i][n]*ge[j][l]*Fe[k][n]+ Fe[j][n]*ge[i][l]*Fe[k][n]);
                        }
            return elasticityTensor;
          }
// compute Jc for Austenite and martensitic variants
           SymmetricTensor<4, dim> get_Jc_A() const
             {

               SymmetricTensor<4,dim> elasticityTensor_A;

                  for (unsigned int n=0; n<dim; ++n)
                    for (unsigned int i=0; i<dim; ++i)
                      for (unsigned int j=0; j<dim; ++j)
                        for (unsigned int k=0; k<dim; ++k)
                          for (unsigned int l=0; l<dim; ++l)
                          {
                            elasticityTensor_A[i][j][k][l] +=
                               lambda_A[n]*Fe[i][n]*Fe[j][n]*Fe[k][n]*Fe[l][n]+
                                   mu_A[n]*(Fe[i][n]*Fe[j][n]*ge[k][l]+ ge[i][j]*Fe[k][n]*Fe[l][n])+
                                   nu_A[n]*(Fe[i][n]*ge[j][k]*Fe[l][n]+ Fe[j][n]*ge[i][k]*Fe[l][n]+
                                          Fe[i][n]*ge[j][l]*Fe[k][n]+ Fe[j][n]*ge[i][l]*Fe[k][n]);
                          }
              return elasticityTensor_A;
            }


                SymmetricTensor<4, dim> get_Jc_M1() const
                {

                  SymmetricTensor<4,dim> elasticityTensor_M1;

                     for (unsigned int n=0; n<dim; ++n)
                       for (unsigned int i=0; i<dim; ++i)
                         for (unsigned int j=0; j<dim; ++j)
                           for (unsigned int k=0; k<dim; ++k)
                             for (unsigned int l=0; l<dim; ++l)
                             {
                               elasticityTensor_M1[i][j][k][l] +=
                                  lambda_M1[n]*Fe[i][n]*Fe[j][n]*Fe[k][n]*Fe[l][n]+
                                      mu_M1[n]*(Fe[i][n]*Fe[j][n]*ge[k][l]+ ge[i][j]*Fe[k][n]*Fe[l][n])+
                                      nu_M1[n]*(Fe[i][n]*ge[j][k]*Fe[l][n]+ Fe[j][n]*ge[i][k]*Fe[l][n]+
                                             Fe[i][n]*ge[j][l]*Fe[k][n]+ Fe[j][n]*ge[i][l]*Fe[k][n]);
                             }
                 return elasticityTensor_M1;
               }


                   SymmetricTensor<4, dim> get_Jc_M2() const
                   {

                     SymmetricTensor<4,dim> elasticityTensor_M2;

                        for (unsigned int n=0; n<dim; ++n)
                          for (unsigned int i=0; i<dim; ++i)
                            for (unsigned int j=0; j<dim; ++j)
                              for (unsigned int k=0; k<dim; ++k)
                                for (unsigned int l=0; l<dim; ++l)
                                {
                                  elasticityTensor_M2[i][j][k][l] +=
                                     lambda_M2[n]*Fe[i][n]*Fe[j][n]*Fe[k][n]*Fe[l][n]+
                                         mu_M2[n]*(Fe[i][n]*Fe[j][n]*ge[k][l]+ ge[i][j]*Fe[k][n]*Fe[l][n])+
                                         nu_M2[n]*(Fe[i][n]*ge[j][k]*Fe[l][n]+ Fe[j][n]*ge[i][k]*Fe[l][n]+
                                                Fe[i][n]*ge[j][l]*Fe[k][n]+ Fe[j][n]*ge[i][l]*Fe[k][n]);
                                }
                    return elasticityTensor_M2;
                  }


                      SymmetricTensor<4, dim> get_Jc_M3() const
                      {

                        SymmetricTensor<4,dim> elasticityTensor_M3;

                           for (unsigned int n=0; n<dim; ++n)
                             for (unsigned int i=0; i<dim; ++i)
                               for (unsigned int j=0; j<dim; ++j)
                                 for (unsigned int k=0; k<dim; ++k)
                                   for (unsigned int l=0; l<dim; ++l)
                                   {
                                     elasticityTensor_M3[i][j][k][l] +=
                                        lambda_M3[n]*Fe[i][n]*Fe[j][n]*Fe[k][n]*Fe[l][n]+
                                            mu_M3[n]*(Fe[i][n]*Fe[j][n]*ge[k][l]+ ge[i][j]*Fe[k][n]*Fe[l][n])+
                                            nu_M3[n]*(Fe[i][n]*ge[j][k]*Fe[l][n]+ Fe[j][n]*ge[i][k]*Fe[l][n]+
                                                   Fe[i][n]*ge[j][l]*Fe[k][n]+ Fe[j][n]*ge[i][l]*Fe[k][n]);
                                   }
                       return elasticityTensor_M3;
                     }

////////Isotropic Material
//           // compute the Kirchhoff stress
//                  SymmetricTensor<2, dim> get_tau() const
//                  {
//                   SymmetricTensor<2, dim> kirchhoff_stress = lambda_iso*I1*ge +
//                                mu_iso*symmetrize(Tensor<2, dim>(ge) * Tensor<2, dim>(ge))-mu_iso*ge;
//                  return kirchhoff_stress;
//                  }
//
//          // compute the modulus J*d_ijkl
//                  SymmetricTensor<4, dim> get_Jc() const
//                  {
//                  SymmetricTensor<4,dim> elasticityTensor;
//                     for (unsigned int i=0; i<dim; ++i)
//                       for (unsigned int j=0; j<dim; ++j)
//                         for (unsigned int k=0; k<dim; ++k)
//                           for (unsigned int l=0; l<dim; ++l)
//                             elasticityTensor[i][j][k][l] =
//                                lambda_iso*ge[i][j]*ge[k][l]+mu_iso*(ge[i][l]*ge[j][k]+ge[i][k]*ge[j][l]);
//                     return elasticityTensor;
//                  }

      // compute the driving force excluding the transformational work
       double get_driving_force_noStress () const
       {
        return  det_F*(delta_psi0+A0*(1-2*c_total));
       }

        double get_det_F() const
       {
         return det_F;
       }
// compute the threshhold related terms for the calibration of the instability criteria
       double get_threshold_c1() const
       {
           const double k1=kd1+(kr1-kd1)*c1;
           const double k3=kd3+(kr3-kd3)*c1;

           SymmetricTensor<2, dim>kirchhoff_stress= get_tau();
           const double k_c1=k1*(kirchhoff_stress[0][0]+kirchhoff_stress[1][1])+k3*kirchhoff_stress[2][2];
        return k_c1;
       }

       double get_threshold_c2() const
       {
           const double k1=kd1+(kr1-kd1)*c2;
           const double k3=kd3+(kr3-kd3)*c2;

           SymmetricTensor<2, dim>kirchhoff_stress= get_tau();
           const double k_c2=k1*(kirchhoff_stress[0][0]+kirchhoff_stress[2][2])+k3*kirchhoff_stress[1][1];
        return k_c2;
       }

       double get_threshold_c3() const
       {
           const double k1=kd1+(kr1-kd1)*c3;
           const double k3=kd3+(kr3-kd3)*c3;

           SymmetricTensor<2, dim>kirchhoff_stress= get_tau();
           const double k_c3=k1*(kirchhoff_stress[1][1]+kirchhoff_stress[2][2])+k3*kirchhoff_stress[0][0];
        return k_c3;
       }


     protected:
       double det_F;
       double I1;
       Tensor<2, dim> Fe;
       SymmetricTensor<2, dim> ge;
       SymmetricTensor<2, dim> Be;

       SymmetricTensor<2, dim> Ge;
       Tensor<2, dim> Ee;
       Tensor<2, dim> FeEe;
       Tensor<2, dim> EeEe;

       Vector<double> C_A,C_M1,C_M2,C_M3;
       Vector<double> lambda_A, lambda_M1,lambda_M2,lambda_M3, lambda;
       Vector<double> mu_A, mu_M1,mu_M2,mu_M3,mu;
       Vector<double> nu_A, nu_M1,nu_M2,nu_M3,nu;

       double lambda_A_iso,lambda_M_iso,lambda_iso;
       double mu_A_iso,mu_M_iso,mu_iso;


       double A0;
       double delta_psi0;
       double ki00;
       double c_total;
       double c0,c1, c2, c3;
       double kd1,kr1,kd3,kr3;



     };
//////////////////////////////////////////////////////////////////////////////////////////
// updates the quadrature point history

  template <int dim>
    class PointHistory
    {
    public:
      PointHistory()
        :
        material(NULL),
        F_inv(StandardTensors<dim>::I),
        tau(SymmetricTensor<2, dim>()),
        Jc(SymmetricTensor<4, dim>())
      {}

      virtual ~PointHistory()
      {
        delete material;
        material = NULL;
      }

      void setup_lqp (const Parameters::AllParameters &parameters)
      {
        material = new Material_Constitutive<dim>(parameters.lambdaA,
                parameters.muA,parameters.lambdaM,parameters.muM,parameters.A, parameters.delta_psi);

        update_values(Tensor<2, dim>(), double (),double (),double (), double(), double(), Point<dim>() );
      }

      void update_values (const Tensor<2, dim> &Grad_u_n, const double c1, const double c2, const double c3,
                          const double dt, const double landa, const Point<dim>  q_point )
      {

// Total deformation gradient
        F = (Tensor<2, dim>(StandardTensors<dim>::I) +  Grad_u_n);
// Transformation strain
        Tensor<2, dim> eps_t1;
          eps_t1[0][0] = 0.1753;
         eps_t1[1][1] = 0.1753;
         eps_t1[2][2] = -0.447;

        Tensor<2, dim> eps_t2;
         eps_t2[0][0] = 0.1753;
         eps_t2[1][1] =  -0.447;
         eps_t2[2][2] = 0.1753;

        Tensor<2, dim> eps_t3;
         eps_t3[0][0] = -0.447;
         eps_t3[1][1] = 0.1753;
         eps_t3[2][2] = 0.1753;

        Ft = Tensor<2, dim>(StandardTensors<dim>::I) + eps_t1*c1+eps_t2*c2+eps_t3*c3; // Transformation deformation gradient
        Fe = F * invert(Ft); // Elastic deformation gradient
        material->update_material_data(F, Fe, c1,c2,c3);

        E=0.5*(symmetrize(transpose(F)*F)-StandardTensors<dim>::I); //Total lagrangian strain
        Ee=0.5*(symmetrize(transpose(Fe)*Fe)-StandardTensors<dim>::I); //Elastic lagrangian strain
        F_inv = invert(F);
        F_inv_tr = transpose(F_inv);
        tau = material->get_tau(); // extracting kirchhoff stress
        Jc = material->get_Jc();  // extracting Jacobian
//        Jc_A = material->get_Jc_A();
//        Jc_M1 = material->get_Jc_M1();
//        Jc_M2 = material->get_Jc_M2();
//        Jc_M3 = material->get_Jc_M3();
        driving_force_noStress = material->get_driving_force_noStress(); // extracting driving force with no stress
        k_c1=material->get_threshold_c1();
        k_c2=material->get_threshold_c2();
        k_c3=material->get_threshold_c3();

        const Tensor<2, dim> temp_tensor = F_inv * Tensor<2, dim>(tau);
        const Tensor<2, dim> temp_tensor1 = temp_tensor * Fe;

// driving force from austenite (0) to each martensitic variant (1,2,3) and between the variants.
//The last term can be included if you want to consider change in elasitc property due to PT.
        X10 = scalar_product(temp_tensor1, eps_t1) - driving_force_noStress - k_c1 /*- 0.5*Ee*(Jc_M1-Jc_A)*Ee*/;
        X20 = scalar_product(temp_tensor1, eps_t2) - driving_force_noStress - k_c2 /*- 0.5*Ee*(Jc_M2-Jc_A)*Ee*/;
        X30 = scalar_product(temp_tensor1, eps_t3) - driving_force_noStress - k_c3 /*- 0.5*Ee*(Jc_M3-Jc_A)*Ee*/;

        X21 = scalar_product(temp_tensor1, (eps_t2-eps_t1))/*- 0.5*Ee*(Jc_M2-Jc_M1)*Ee*/;
        X31 = scalar_product(temp_tensor1, (eps_t3-eps_t1))/*- 0.5*Ee*(Jc_M3-Jc_M1)*Ee*/;
        X32 = scalar_product(temp_tensor1, (eps_t3-eps_t2))/*- 0.5*Ee*(Jc_M3-Jc_M2)*Ee*/;

        const double c0=1-c1-c2-c3;

// Implementation of the constraints on the kinetic equation

        if ((X10>0 && c1<1 && c0>0) || (X10<0 && c1>0 && c0<1))
            dc10 = dt * landa * X10;

        if ((X20>0 && c2<1 && c0>0) || (X20<0 && c2>0 && c0<1))
            dc20 = dt * landa * X20;

        if ((X30>0 && c3<1 && c0>0) || (X30<0 && c3>0 && c0<1))
            dc30 = dt * landa * X30;



        if  ((X12>0 && c1<1 && c2>0) || (X12<0 && c1>0 && c2<1))
            dc12 = dt * landa * X12;

        if  ((X13>0 && c1<1 && c3>0) || (X13<0 && c1>0 && c3<1))
            dc13 = dt * landa * X13;

        if  ((X23>0 && c2<1 && c3>0) || (X23<0 && c2>0 && c3<1))
            dc23 = dt * landa * X23;


        if(dc10>0){
            if (c0-dc10<0 || c1+dc10>1)
                dc10=std::min(1-c1,c0);
        }
        else if (dc10<0){
            if (c1-abs(dc10)<0 || c0+abs(dc10)>1)
                dc10=-std::min(c1,1-c0);
        }

        if(dc20>0){
            if (c0-dc20<0 || c2+dc20>1)
                dc20=std::min(1-c2,c0);
        }
        else if (dc20<0){
            if (c2-abs(dc20)<0 || c0+abs(dc20)>1)
                dc20=-std::min(c2,1-c0);
        }

        if(dc30>0){
            if (c0-dc30<0 || c3+dc30>1)
                dc30=std::min(1-c3,c0);
        }
        else if (dc30<0){
            if (c3-abs(dc30)<0 || c0+abs(dc30)>1)
                dc30=-std::min(c3,1-c0);
        }

        if(dc12>0){
            if (c2-dc12<0 || c1+dc12>1)
                dc12=std::min(1-c1,c2);
        }
        else if (dc12<0){
            if (c1-abs(dc12)<0 || c2+abs(dc12)>1)
                dc12=-std::min(c1,1-c2);
        }

        if(dc13>0){
            if (c3-dc13<0 || c1+dc13>1)
                dc13=std::min(1-c1,c3);
        }
        else if (dc13<0){
            if (c1-abs(dc13)<0 || c3+abs(dc13)>1)
                dc13=-std::min(c1,1-c3);
        }

        if(dc23>0){
            if (c3-dc23<0 || c2+dc23>1)
                dc23=std::min(1-c2,c3);
        }
        else if (dc23<0){
            if (c2-abs(dc23)<0 || c3+abs(dc23)>1)
                dc23=-std::min(c2,1-c3);
        }

        const double dci0=dc10+dc20+dc30;

               if(dci0>0){
                   if(c0-dci0<0){
                    dc10=0;
                       dc20=0;
                       dc30=0;
                   }
               }
               else if (dci0<0){
                   if(c0+abs(dci0)>1){
                       dc10=0;
                       dc20=0;
                       dc30=0;

                   }
               }

        dc21=-dc12;
        dc31=-dc13;
        dc32=-dc23;

        dc1=dc10+dc12+dc13;
        dc2=dc20+dc21+dc23;
        dc3=dc30+dc31+dc32;

// Excluding PT for the top and bottom of the sample to have pure elastic deformation
        if (q_point[2]<0.156 || q_point[2]>2.844){
          dc1=0;
          dc2=0;
          dc3=0;
        }

        Assert(determinant(F_inv) > 0, ExcInternalError());
      }


      const Tensor<2, dim> &get_F() const
      {
        return F;
      }

      double get_det_F() const
      {
        return material->get_det_F();
      }

      const Tensor<2, dim> &get_F_inv() const
      {
        return F_inv;
      }
      const Tensor<2, dim> &get_E() const
      {
        return E;
      }

      const Tensor<2, dim> &get_F_inv_tr() const
      {
        return F_inv_tr;
      }

      const SymmetricTensor<2, dim> &get_tau() const
      {
        return tau;
      }

      const SymmetricTensor<4, dim> &get_Jc() const
      {
        return Jc;
      }

      double get_update_c1() const
      {
        return dc1;
      }
      double get_update_c2() const
      {
        return dc2;
      }
      double get_update_c3() const
      {
        return dc3;
      }

    private:
      Material_Constitutive<dim> *material;
      Tensor<2, dim> F;
      Tensor<2, dim> F_inv;
      Tensor<2, dim> F_inv_tr;
      Tensor<2, dim> Ft;
      Tensor<2, dim> E;
      SymmetricTensor<2, dim> Ee;
      SymmetricTensor<2, dim> tau;
      SymmetricTensor<4, dim> Jc, Jc_A,Jc_M1,Jc_M2,Jc_M3;
      Tensor<2, dim> Fe;
      double driving_force_noStress;
      double X10,X12,X13,X20,X21,X23,X30,X31,X32;
      double dc10,dc12,dc13,dc20,dc21,dc23,dc30,dc31,dc32;
      double dc1,dc2,dc3;
      double k_c1,k_c2,k_c3;

    };
///////////////////////////////////////////////////////////////

  template <int dim>
  class Solid
  {
  public:
    Solid(const std::string &input_file);

    virtual
    ~Solid();

    void
    run();

  private:

    void    make_grid();
    void    system_setup();
    void    assemble_system();
    void    make_constraints(const int &it_nr);
    void    solve_nonlinear_timestep();
    unsigned int    solve();
    void    assemble_system_c();
    void    solve_c();
    void    setup_qph();
    void    update_qph_incremental();
    void    output_results() const;
    void    output_resultant_stress();


    Parameters::AllParameters        parameters;

    Time                             time;  // variable of type class 'Time'
    //TimerOutput                      timer;


    MPI_Comm                         mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    ConditionalOStream               pcout;

    const unsigned int               degree; // degree of polynomial of shape functions
    const FESystem<dim>              fe; // fe object
    DoFHandler<dim>                 dof_handler; // we have used two dof_handler: one for mechanics another for order parameter
    const unsigned int               dofs_per_cell;   // no of dofs per cell for the mechanics problem
    const FEValuesExtractors::Vector   u_fe;
    const QGauss<dim>                qf_cell;  // quadrature points in the cell
    const QGauss<dim - 1>            qf_face;  // quadrature points at the face
    const unsigned int               n_q_points;  // no of quadrature points in the cell
    const unsigned int               n_q_points_f; // no of quadrature points at the face
    ConstraintMatrix                 constraints;  // constraint object

    FE_DGQ<dim>                     history_fe;
    DoFHandler<dim>                    history_dof_handler;

    std::vector<PointHistory<dim> >  quadrature_point_history;

    IndexSet                         locally_owned_dofs;
    IndexSet                         locally_relevant_dofs;


    matrixType                   tangent_matrix;  // tangent stiffenss matrix
    vectorType                system_rhs;  // system right hand side or residual of mechanics problem
    vectorType                solution;  // solution vector for displacement
    vectorType                solution_update; // another vector containing the displacement soln


    const unsigned int               degree_c; // degree of polynomial for c
    FE_Q<dim>                        fe_c;  // fe object for c
    DoFHandler<dim>                  dof_handler_c; //another dof_handler for c
    const unsigned int               dofs_per_cell_c; // dof per c cell
    const QGauss<dim>                qf_cell_c;
    const unsigned int               n_q_points_c;
    ConstraintMatrix                 constraints_c;
    IndexSet                         locally_owned_dofs_c;
    IndexSet                         locally_relevant_dofs_c;

    matrixType                  mass_matrix;
    vectorType                  system_rhs_c1, system_rhs_c2, system_rhs_c3;
    vectorType                  solution_c0, solution_c1, solution_c2, solution_c3;
    vectorType                  old_solution_c1, old_solution_c2, old_solution_c3;
    vectorType                  solution_update_c1, solution_update_c2, solution_update_c3 ;


    Vector<double>                   resultant_cauchy_stress;
    Vector<double>                   resultant_first_piola_stress;
    Vector<double>                   resultant_second_piola_stress;
    Vector<double>                   static_cauchy_stress;
    Vector<double>                   static_first_piola_stress;
    Vector<double>                   static_second_piola_stress;
    Vector<double>                   resultant_lagrangian_strain;
    Vector<double>                   static_lagrangian_strain;
    Vector<double>                   order_parameter;
    Vector<double>                   static_order_parameter;
    bool                             apply_strain;
    double                           load_step;
    double                           load;

  };
/////////////////////////////////////////////////////////
  // defines the initial condition for the order parameter
           template <int dim>
           class InitialValues : public Function<dim>
           {
           public:
             InitialValues (const int &variant,const int &time_step)
              :
               Function<dim>(),
                 variant (variant),
               time_step (time_step)

                {}
             virtual double value(const Point<dim>   &p,
                                   const unsigned int  /*component = 0*/) const;
           private:
             const int variant ;
             const int time_step ;

           };

           template <int dim>
           double InitialValues<dim>::value (const Point<dim>  &p,
                                      const unsigned int /*component*/) const
          {

//               if (pow(p[0]-35,2)+pow(p[1]-10,2)<4)
//                {
//                   if (variant==1)
//                       return 0.01;
//                   else
//                       return 0;
//                }
//               else if (time_step==0)
                   return 0.0;

          }

//////////////////////////////////////////////////////////////////
//Applying displacement on the boundary if periodic condition is absent.
//Displacement on the bpoundary can be alternatively applied through periodic condition as well.
         template <int dim>
            class BoundaryDisplacement :  public Function<dim>
            {
            public:
              BoundaryDisplacement (const double direction, const int timestep);
              virtual
              void
              vector_value (const Point<dim> &p,
                            Vector<double>   &values) const;
              virtual
              void
              vector_value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Vector<double> >   &value_list) const;
            private:
              const double direction;
              const int timestep;
            };
            template <int dim>
            BoundaryDisplacement<dim>::BoundaryDisplacement (const double direction, const int timestep)
              :
              Function<dim> (dim),
              direction(direction),
              timestep(timestep)
            {}
            template <int dim>
            inline
            void
            BoundaryDisplacement<dim>::vector_value (const Point<dim> &/*p*/,
                                          Vector<double>   &values) const
            {
              Assert (values.size() == dim,
                      ExcDimensionMismatch (values.size(), dim));

              values = 0.0;
              //values(direction)=-5e-4;

              if(timestep<70)
                  values(direction)=-3e-3;
              else if(timestep%10==0)
                  values(direction)=-3e-3;
              else
                  values(direction)=0.0;




            }
            template <int dim>
            void
            BoundaryDisplacement<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                               std::vector<Vector<double> >   &value_list) const
            {
              const unsigned int n_ooints = points.size();
              Assert (value_list.size() == n_ooints,
                      ExcDimensionMismatch (value_list.size(), n_ooints));
              for (unsigned int p=0; p<n_ooints; ++p)
                  BoundaryDisplacement<dim>::vector_value (points[p],
                                              value_list[p]);
            }

///////////////////////////////////////////////////////////
// constructor
  template <int dim>
  Solid<dim>::Solid(const std::string &input_file)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),

    parameters(input_file),
    time(parameters.end_time, parameters.delta_t),

    pcout (std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator)
           == 0)),
//    timer(mpi_communicator,
//          pcout,
//          TimerOutput::summary,
//          TimerOutput::wall_times),

    degree(parameters.poly_degree),
    fe(FE_Q<dim>(parameters.poly_degree), dim), // displacement
    dof_handler(triangulation),
    dofs_per_cell (fe.dofs_per_cell),
    u_fe(0),
    qf_cell(parameters.quad_order),
    qf_face(parameters.quad_order),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size()),

    degree_c(parameters.poly_degree),
    fe_c (parameters.poly_degree),
    dof_handler_c (triangulation),
    dofs_per_cell_c(fe_c.dofs_per_cell),
    qf_cell_c(parameters.quad_order),
    n_q_points_c (qf_cell_c.size()),

    history_dof_handler (triangulation),
    history_fe (parameters.poly_degree),
    apply_strain(false),
    load_step(1),
    load(0.0)
  {}

//destructor
  template <int dim>
  Solid<dim>::~Solid()
  {
    dof_handler.clear();
    dof_handler_c.clear();
  }


///////////////////////////////////////////////////////
//creating geometry and discritization
  template <int dim>
  void Solid<dim>::make_grid()
  {

    std::vector< unsigned int > repetitions(dim, 30);
    if (dim == 3)
    repetitions[dim-2] = 10;
    repetitions[dim-3] = 5;

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              Point<dim>(0.0, 0.0, 0.0),
                                              Point<dim>(0.5, 1.0, 3.0),
                                                 true);
// sample code to import mesh from other softwares such as Abaqus
/*
      GridIn<3> grid_in;
      grid_in.attach_triangulation (triangulation);

      pcout << "Reading grid file:" << std::endl;
      std::ifstream input_file("grid_input.inp");
      grid_in.read_abaqus (input_file);
      pcout << "grid_input file was read succesfully" << std::endl;

          typename Triangulation<dim>::face_iterator
      face = triangulation.begin_face(),
      endface = triangulation.end_face();
      for (; face!=endface; ++face)
          if (face->at_boundary())
            if (face->boundary_id() == 0)
              {
                const Point<dim> face_center (face->center());

                if      (std::abs(face_center[0])<1e-8)
                  face->set_boundary_id(0);
                else if (std::abs(face_center[0]-40.0)<1e-8)
                  face->set_boundary_id(1);
                else if (std::abs(face_center[1])<1e-8)
                  face->set_boundary_id(2);
                else if (std::abs(face_center[1]-20.0)<1e-8)
                  face->set_boundary_id(3);
                else if (std::abs(face_center[2])<1e-8)
                  face->set_boundary_id(4);
                else if (std::abs(face_center[2]-20.0)<1e-8)
                  face->set_boundary_id(5);
                else
                  // triangulation says it
                  // is on the boundary,
                  // but we could not find
                  // on which boundary.
                   Assert (false, ExcInternalError());

              }
*/


// //Applying periodic conditions on the mesh
//      std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator> >
//           periodicity_vector;
//     GridTools::collect_periodic_faces(triangulation,
//                                            /*b_id1*/ 0,
//                                            /*b_id2*/ 1,
//                                            /*direction*/ 0,
//                                            periodicity_vector);
//     GridTools::collect_periodic_faces(triangulation,
//                                           /*b_id1*/ 2,
//                                           /*b_id2*/ 3,
//                                           /*direction*/ 1,
//                                           periodicity_vector );
//     GridTools::collect_periodic_faces(triangulation,
//                                            /*b_id1*/ 4,
//                                            /*b_id2*/ 5,
//                                            /*direction*/ 2,
//                                            periodicity_vector);
//triangulation.add_periodicity(periodicity_vector);


//triangulation.refine_global (parameters.refinement);

  }
///////////////////////////////////////////////////////
// initializing the system matrix and system vectors
  template <int dim>
  void Solid<dim>::system_setup()
  {
    dof_handler.distribute_dofs(fe);
    dof_handler_c.distribute_dofs (fe_c);
    history_dof_handler.distribute_dofs (history_fe);

    const unsigned int n_dofs = dof_handler.n_dofs(),
                        n_dofs_c  = dof_handler_c.n_dofs();

         pcout     << "   Number of active cells: "
                       << triangulation.n_active_cells()
                       << std::endl
                       << "   Total number of cells: "
                       << triangulation.n_cells()
                       << std::endl
                          << "   Number of degrees of freedom: "
                       << n_dofs + n_dofs_c
                       << " (" << n_dofs << '+' << n_dofs_c << ')'
                       << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler  ,  locally_relevant_dofs);

    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);

    {
//      DoFTools::make_periodicity_constraints(dof_handler,
//                                             /*b_id*/ 0,
//                                             /*b_id*/ 1,
//                                             /*direction*/ 0,
//                                             constraints);
//      DoFTools::make_periodicity_constraints(dof_handler,
//                                             /*b_id*/ 2,
//                                             /*b_id*/ 3,
//                                             /*direction*/ 1,
//                                             constraints);
//      DoFTools::make_periodicity_constraints(dof_handler,
//                                               /*b_id*/ 4,
//                                               /*b_id*/ 5,
//                                               /*direction*/ 2,
//                                               constraints);

   }
   constraints.close ();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern (dsp,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);

    tangent_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);

    solution.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    solution_update.reinit(locally_owned_dofs,
                           locally_relevant_dofs,
                           mpi_communicator);

    system_rhs.reinit(locally_owned_dofs,
                      mpi_communicator);


    locally_owned_dofs_c = dof_handler_c.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler_c  ,  locally_relevant_dofs_c);

    constraints_c.clear ();
    constraints_c.reinit (locally_relevant_dofs_c);
    {

//        DoFTools::make_periodicity_constraints(dof_handler_c,
//                                               /*b_id*/ 0,
//                                               /*b_id*/ 1,
//                                               /*direction*/ 0,
//                                               constraints_c);
//        DoFTools::make_periodicity_constraints(dof_handler_c,
//                                                 /*b_id*/ 2,
//                                               /*b_id*/ 3,
//                                               /*direction*/ 1,
//                                               constraints_c);
//        DoFTools::make_periodicity_constraints(dof_handler_c,
//                                               /*b_id*/ 4,
//                                               /*b_id*/ 5,
//                                               /*direction*/ 2,
//                                               constraints_c);

    }
    constraints_c.close ();

    DynamicSparsityPattern dsp_c(locally_relevant_dofs_c);
    DoFTools::make_sparsity_pattern (dof_handler_c, dsp_c, constraints_c, true);
    SparsityTools::distribute_sparsity_pattern (dsp_c,
                                                dof_handler_c.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs_c);

    mass_matrix.reinit (locally_owned_dofs_c, locally_owned_dofs_c, dsp_c, mpi_communicator);


    solution_c0.reinit(locally_owned_dofs_c, locally_relevant_dofs_c, mpi_communicator);
    solution_c1.reinit(locally_owned_dofs_c, locally_relevant_dofs_c, mpi_communicator);
    solution_c2.reinit(locally_owned_dofs_c, locally_relevant_dofs_c, mpi_communicator);
    solution_c3.reinit(locally_owned_dofs_c, locally_relevant_dofs_c, mpi_communicator);


    system_rhs_c1.reinit(locally_owned_dofs_c, mpi_communicator);
    system_rhs_c2.reinit(locally_owned_dofs_c, mpi_communicator);
    system_rhs_c3.reinit(locally_owned_dofs_c, mpi_communicator);

    resultant_cauchy_stress.reinit(100000);
    static_cauchy_stress.reinit   (100000);
    resultant_first_piola_stress.reinit(100000);
    static_first_piola_stress.reinit   (100000);
    resultant_second_piola_stress.reinit(100000);
    static_second_piola_stress.reinit   (100000);
    resultant_lagrangian_strain.reinit(100000);
    static_lagrangian_strain.reinit(100000);
    order_parameter.reinit(100000);
    static_order_parameter.reinit(100000);

   setup_qph();


  }

//////////////////////////////////
// Applying displacement constraints on the system
  template <int dim>
  void Solid<dim>::make_constraints(const int &it_nr)
  {
    if (it_nr > 1)
      return;

    constraints.clear();
    constraints.reinit (locally_relevant_dofs);

    const bool apply_dirichlet_bc = (it_nr == 0);
    const int  timestep = time.get_timestep();

    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    const FEValuesExtractors::Scalar z_displacement(2);

// // Fixing points or lines

//        const double tol_boundary = 0.01;
//        typename DoFHandler<dim>::active_cell_iterator
//         cell = dof_handler.begin_active(),
//        endc = dof_handler.end();
//        for (; cell!=endc; ++cell)
//            if (cell->is_locally_owned())
//              {
//                  for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
//
//                  if     ((std::abs(cell->vertex(v)[0] - 0.5) < tol_boundary) &&
//                          (std::abs(cell->vertex(v)[1] - 0.5) < tol_boundary) &&
//                          (std::abs(cell->vertex(v)[2] - 1.0) < tol_boundary))
//                       {
//                            constraints.add_line(cell->vertex_dof_index(v, 0));
//                            constraints.add_line(cell->vertex_dof_index(v, 1));
//                            constraints.add_line(cell->vertex_dof_index(v, 2));
//                       }
//            else if     (/*(std::abs(cell->vertex(v)[0] - 0.0) < tol_boundary) &&*/
//                         (std::abs(cell->vertex(v)[1] - 0.0) < tol_boundary) &&
//                         (std::abs(cell->vertex(v)[2] - 0.0) < tol_boundary))
//                      {
//                         // constraints.add_line(cell->vertex_dof_index(v, 0));
//                         constraints.add_line(cell->vertex_dof_index(v, 1));
//                         //constraints.add_line(cell->vertex_dof_index(v, 2));
//                      }
//
//           else if      ((std::abs(cell->vertex(v)[0] - 0.0) < tol_boundary) &&
//                         /*(std::abs(cell->vertex(v)[1] - 0.0) < tol_boundary) &&*/
//                         (std::abs(cell->vertex(v)[2] - 0.0) < tol_boundary))
//                      {
//                         constraints.add_line(cell->vertex_dof_index(v, 0));
//                        //constraints.add_line(cell->vertex_dof_index(v, 1));
//                        //constraints.add_line(cell->vertex_dof_index(v, 2));
//                     }
//           else if      ((std::abs(cell->vertex(v)[0] - 0.0) < tol_boundary) &&
//                         (std::abs(cell->vertex(v)[1] - 1.0) < tol_boundary) &&
//                         (std::abs(cell->vertex(v)[2] - 0.0) < tol_boundary))
//                     {
//                        constraints.add_line(cell->vertex_dof_index(v, 0));
//                         //constraints.add_line(cell->vertex_dof_index(v, 1));
//                        //constraints.add_line(cell->vertex_dof_index(v, 2));
//                     }
//              }

// Fixing external surfaces
                    {
                         const int boundary_id = 0;

                    if (apply_dirichlet_bc == true)
                      VectorTools::interpolate_boundary_values(dof_handler,
                                                               boundary_id,
                                                               ZeroFunction<dim>(dim),
                                                               constraints,
                                                               (fe.component_mask(x_displacement)));
                    else
                      VectorTools::interpolate_boundary_values(dof_handler,
                                                               boundary_id,
                                                               ZeroFunction<dim>(dim),
                                                               constraints,
                                                               (fe.component_mask(x_displacement)));
                    }
//
//                    {
//                        const int boundary_id = 1;
//
//                         if (apply_dirichlet_bc == true)
//                          VectorTools::interpolate_boundary_values(dof_handler,
//                                                                   boundary_id,
//                                                                      BoundaryDisplacement<dim>(0, timestep),
//                                                                   constraints,
//                                                                   fe.component_mask(x_displacement));
//                        else
//                          VectorTools::interpolate_boundary_values(dof_handler,
//                                                                   boundary_id,
//                                                                   ZeroFunction<dim>(dim),
//                                                                   constraints,
//                                                                   fe.component_mask(x_displacement));
//                   }

//                    {
//                          const int boundary_id = 2;
//
//                     if (apply_dirichlet_bc == true)
//                       VectorTools::interpolate_boundary_values(dof_handler,
//                                                                boundary_id,
//                                                                ZeroFunction<dim>(dim),
//                                                               constraints,
//                                                                fe.component_mask(y_displacement));
//                     else
//                       VectorTools::interpolate_boundary_values(dof_handler,
//                                                                boundary_id,
//                                                                ZeroFunction<dim>(dim),
//                                                                constraints,
//                                                                fe.component_mask(y_displacement));
//                     }

//                    {
//                           const int boundary_id = 3;
//
//                      if (apply_dirichlet_bc == true)
//                        VectorTools::interpolate_boundary_values(dof_handler,
//                                                                 boundary_id,
//                                                               BoundaryDisplacement<dim>(1, timestep),
//                                                                constraints,
//                                                                 fe.component_mask(y_displacement));
//                      else
//                        VectorTools::interpolate_boundary_values(dof_handler,
//                                                                 boundary_id,
//                                                                 ZeroFunction<dim>(dim),
//                                                                 constraints,
//                                                                 fe.component_mask(y_displacement));
//                      }

                    {

                        const int boundary_id = 4;

                          if (apply_dirichlet_bc == true)
                            VectorTools::interpolate_boundary_values(dof_handler,
                                                                     boundary_id,
                                                                     ZeroFunction<dim>(dim),
                                                                     constraints,
                                                                     fe.component_mask(x_displacement)
                                                                   |
                                                                   fe.component_mask(y_displacement)
                                                                   |
                                                                   fe.component_mask(z_displacement));
                          else
                            VectorTools::interpolate_boundary_values(dof_handler,
                                                                     boundary_id,
                                                                     ZeroFunction<dim>(dim),
                                                                     constraints,
                                                                   fe.component_mask(x_displacement)
                                                                   |
                                                                   fe.component_mask(y_displacement)
                                                                   |
                                                                   fe.component_mask(z_displacement));
                     }
                    {

                        const int boundary_id = 5;

                          if (apply_dirichlet_bc == true)
                            VectorTools::interpolate_boundary_values(dof_handler,
                                                                     boundary_id,
                                                                   BoundaryDisplacement<dim>(2, timestep),
                                                                     constraints,
                                                                     fe.component_mask(z_displacement));
                          else
                            VectorTools::interpolate_boundary_values(dof_handler,
                                                                     boundary_id,
                                                                     ZeroFunction<dim>(dim),
                                                                     constraints,
                                                                     fe.component_mask(z_displacement));
                     }
                   {

                       const int boundary_id = 5;

                         if (apply_dirichlet_bc == true)
                           VectorTools::interpolate_boundary_values(dof_handler,
                                                                    boundary_id,
                                                                   ZeroFunction<dim>(dim),
                                                                    constraints,
                                                                    fe.component_mask(x_displacement)
                                                                   |
                                                                   fe.component_mask(y_displacement));
                         else
                           VectorTools::interpolate_boundary_values(dof_handler,
                                                                    boundary_id,
                                                                    ZeroFunction<dim>(dim),
                                                                    constraints,
                                                                    fe.component_mask(x_displacement)
                                                                   |
                                                                   fe.component_mask(y_displacement));
                    }

// In the case of periodic conditions these lines shold be uncommented.

//{
//        DoFTools::make_periodicity_constraints(dof_handler,
//                                              /*b_id*/ 0,
//                                              /*b_id*/ 1,
//                                              /*direction*/ 0,
//                                              constraints);
//        DoFTools::make_periodicity_constraints(dof_handler,
//                                              /*b_id*/ 2,
//                                              /*b_id*/ 3,
//                                              /*direction*/ 1,
//                                                 constraints);
//       DoFTools::make_periodicity_constraints(dof_handler,
//                                               /*b_id*/ 4,
//                                               /*b_id*/ 5,
//                                               /*direction*/ 2,
//                                               constraints);
//      }
//     constraints.close();

//     // Applying boundary displacement when having periodic conditions
/*
      {
         IndexSet selected_dofs_x;
         std::set< types::boundary_id > boundary_ids_x= std::set<types::boundary_id>();
                 boundary_ids_x.insert(0);

         DoFTools::extract_boundary_dofs(dof_handler,
                                        fe.component_mask(x_displacement),
                                             selected_dofs_x,
                                             boundary_ids_x);
         unsigned int nb_dofs_face_x = selected_dofs_x.n_elements();
         IndexSet::ElementIterator dofs_x = selected_dofs_x.begin();

         double relative_displacement_x;

         if(timestep<80)
         relative_displacement_x = 5e-4;
         else if(timestep%100==0)
         relative_displacement_x = 5e-4;
         else
         relative_displacement_x = 0.0;

         for(unsigned int i = 0; i < nb_dofs_face_x; i++)
         {
          constraints.add_line (*dofs_x);
           constraints.set_inhomogeneity(*dofs_x, (apply_dirichlet_bc ? relative_displacement_x : 0.0));
             dofs_x++;
         }
       }


     {
         IndexSet selected_dofs_y;
         std::set< types::boundary_id > boundary_ids_y= std::set<types::boundary_id>();
                 boundary_ids_y.insert(2);

         DoFTools::extract_boundary_dofs(dof_handler,
                                        fe.component_mask(y_displacement),
                                           selected_dofs_y,
                                           boundary_ids_y);
         unsigned int nb_dofs_face_y = selected_dofs_y.n_elements();
         IndexSet::ElementIterator dofs_y = selected_dofs_y.begin();

         double relative_displacement_y = 0.0;

         if(timestep<80)
          relative_displacement_y = 5e-4;
         else if(timestep%100==0)
          relative_displacement_y = 5e-4;
         else
          relative_displacement_y = 0.0;


         for(unsigned int i = 0; i < nb_dofs_face_y; i++)
         {
           constraints.add_line (*dofs_y);
           constraints.set_inhomogeneity(*dofs_y,(apply_dirichlet_bc ? relative_displacement_y : 0.0) );
             dofs_y++;
         }

       }

     {
       IndexSet selected_dofs_z;
       std::set< types::boundary_id > boundary_ids_z= std::set<types::boundary_id>();
               boundary_ids_z.insert(4);

       DoFTools::extract_boundary_dofs(dof_handler,
                                      fe.component_mask(z_displacement),
                                           selected_dofs_z,
                                           boundary_ids_z);
       unsigned int nb_dofs_face_z = selected_dofs_z.n_elements();
       IndexSet::ElementIterator dofs_z = selected_dofs_z.begin();

       double relative_displacement_z;
       relative_displacement_z = 0.0;

       if(timestep<75)
        relative_displacement_z = 10e-4;
       else if(timestep%10==0)
        relative_displacement_z = 10e-4;
       else
        relative_displacement_z = 0.0;


       for(unsigned int i = 0; i < nb_dofs_face_z; i++)
       {
         constraints.add_line (*dofs_z);
           constraints.set_inhomogeneity(*dofs_z,(apply_dirichlet_bc ? relative_displacement_z : 0.0));
           dofs_z++;
       }
    }
*/
    constraints.close();

  }

/////////////////////////////////////////////////////////////////////////////////////
// Assempling system matrix and system RHS for the mechanical equilibrium equation
  template <int dim>
  void Solid<dim>::assemble_system ()
  {
    tangent_matrix = 0;
    system_rhs = 0;

   FEValues<dim> fe_values (fe, qf_cell,
                            update_values   | update_gradients |
                            update_quadrature_points | update_JxW_values);


   FEFaceValues<dim> fe_face_values (fe, qf_face,
                                     update_values         | update_quadrature_points  |
                                     update_normal_vectors | update_JxW_values);

   FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
   Vector<double>       cell_rhs (dofs_per_cell);

   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

   std::vector<double>                    Nx(dofs_per_cell);
   std::vector<Tensor<2, dim> >           grad_Nx(dofs_per_cell);
   std::vector<SymmetricTensor<2, dim> >  symm_grad_Nx(dofs_per_cell);



   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end();
   for (; cell!=endc;  ++cell)
       if (cell->is_locally_owned())
       {
           fe_values.reinit (cell);
           cell_matrix = 0;
        cell_rhs = 0;

       PointHistory<dim> *lqph =
             reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
           {
            const Tensor<2, dim> F_inv = lqph[q_point].get_F_inv();
            const Tensor<2, dim> tau   = lqph[q_point].get_tau();
            const SymmetricTensor<2, dim> symm_tau         = lqph[q_point].get_tau();
            const SymmetricTensor<4, dim> Jc = lqph[q_point].get_Jc();
            const double JxW = fe_values.JxW(q_point);

            for (unsigned int k=0; k<dofs_per_cell; ++k)
               {
                  grad_Nx[k] = fe_values[u_fe].gradient(k, q_point)  * F_inv;
                  symm_grad_Nx[k] = symmetrize(grad_Nx[k]);
               }


          for (unsigned int i=0; i<dofs_per_cell; ++i)
             {
               const unsigned int component_i = fe.system_to_component_index(i).first;

                for (unsigned int j=0; j<dofs_per_cell; ++j)
               {
                     const unsigned int component_j = fe.system_to_component_index(j).first;

                      cell_matrix(i, j) += symm_grad_Nx[i] * Jc // The material contribution:
                                                  * symm_grad_Nx[j] * JxW;
                     if (component_i == component_j) // geometrical stress contribution
                      cell_matrix(i, j) += grad_Nx[i][component_i] * tau
                                                  * grad_Nx[j][component_j] * JxW;
              }

                      cell_rhs(i) -= symm_grad_Nx[i] * symm_tau * JxW;
            }
       }




// Applying force on the external faces

//       // if (time.get_timestep()>3)
//       {
//
//           for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
//
//            if (cell->face(face_number)->at_boundary()
//                    &&
//               ( (cell->face(face_number)->boundary_id() == 0 )
//                     ||
//                 (cell->face(face_number)->boundary_id() == 1 )
//                     ||
//                 (cell->face(face_number)->boundary_id() == 2 )
//                     ||
//                 (cell->face(face_number)->boundary_id() == 3 )))
//             {
//               fe_face_values.reinit (cell, face_number);
//
//               for (unsigned int q_point=0; q_point<n_q_points_f; ++q_point)
//                 {
//                   const Tensor<2, dim> F_inv_tr = lqph[q_point].get_F_inv_tr();
//
//                     const double J= lqph[q_point].get_det_F();
//
//                           double normal_stress;
//                         if (load_step<12)
//                              normal_stress= -load_step * J * (F_inv_tr*fe_face_values.normal_vector(q_point)).norm();
//                         else
//                              normal_stress= -12 * J * (F_inv_tr*fe_face_values.normal_vector(q_point)).norm();
//
//                         const Tensor<1, dim> traction  = normal_stress* fe_face_values.normal_vector(q_point);
//
//
//                  for (unsigned int i=0; i<dofs_per_cell; ++i)
//                   {
//                       const unsigned int
//                     component_i = fe.system_to_component_index(i).first;
//                     cell_rhs(i) += (traction[component_i] *
//                                     fe_face_values.shape_value(i,q_point) *
//                                     fe_face_values.JxW(q_point));
//                   }
//                 }
//               }
//
//            if (cell->face(face_number)->at_boundary()
//                     &&
//             (cell->face(face_number)->boundary_id() == 5)
//
//                 )
//                 {
//                   fe_face_values.reinit (cell, face_number);
//                  for (unsigned int q_point=0; q_point<n_q_points_f; ++q_point)
//                     {
//                      const Tensor<2, dim> F_inv_tr = lqph[q_point].get_F_inv_tr();
//
//                         const double J= lqph[q_point].get_det_F();
//
//                         double normal_stress;
//                         if (load_step<4)
//                            normal_stress= -2*load_step * J * (F_inv_tr*fe_face_values.normal_vector(q_point)).norm();
//                         else
//                        normal_stress= -8.0 * J * (F_inv_tr*fe_face_values.normal_vector(q_point)).norm();
//
//                      const Tensor<1, dim> traction  = normal_stress* fe_face_values.normal_vector(q_point);
//
//
//                       for (unsigned int i=0; i<dofs_per_cell; ++i)
//                       {
//                           const unsigned int
//                         component_i = fe.system_to_component_index(i).first;
//                         cell_rhs(i) += (traction[component_i] *
//                                         fe_face_values.shape_value(i,q_point) *
//                                         fe_face_values.JxW(q_point));
//                       }
//                     }
//                   }
//
//
//              }


      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              tangent_matrix,
                                              system_rhs);

     }

   tangent_matrix.compress (VectorOperation::add);
   system_rhs.compress (VectorOperation::add);

  }

//////////////////////////////////////////////////////
// Assemplying system matrix and RHS for the phase-field kinetic equation
  template <int dim>
  void Solid<dim>::assemble_system_c ()
  {
    mass_matrix = 0;
    system_rhs_c1 = 0;
    system_rhs_c2 = 0;
    system_rhs_c3 = 0;

    FEValues<dim> fe_values_c (fe_c, qf_cell_c,
                             update_values  | update_gradients |
                             update_quadrature_points | update_JxW_values);

    FullMatrix<double>   cell_mass_matrix    (dofs_per_cell_c, dofs_per_cell_c);
    Vector<double>   cell_rhs_c1         (dofs_per_cell_c);
    Vector<double>   cell_rhs_c2         (dofs_per_cell_c);
    Vector<double>   cell_rhs_c3         (dofs_per_cell_c);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_c);

    std::vector<double> phi (dofs_per_cell_c);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_c.begin_active(),
    endc = dof_handler_c.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
        fe_values_c.reinit(cell);

        cell_mass_matrix = 0;
        cell_rhs_c1=0;
        cell_rhs_c2=0;
        cell_rhs_c3=0;

        PointHistory<dim> *lqph =
                      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

        for (unsigned int q=0; q<n_q_points_c; ++q)
          {
            const double dc1 = lqph[q].get_update_c1();
            const double dc2 = lqph[q].get_update_c2();
            const double dc3 = lqph[q].get_update_c3();

            for (unsigned int k=0; k<dofs_per_cell_c; ++k)
              {
                phi[k] = fe_values_c.shape_value (k, q);
                 }

            for (unsigned int i=0; i<dofs_per_cell_c; ++i)
              {
                for (unsigned int j=0; j<dofs_per_cell_c; ++j)
                {
                        cell_mass_matrix(i,j) += phi[i] * phi[j] * fe_values_c.JxW(q);

                }

                cell_rhs_c1(i) +=  dc1 * phi[i] * fe_values_c.JxW (q);
                cell_rhs_c2(i) +=  dc2 * phi[i] * fe_values_c.JxW (q);
                cell_rhs_c3(i) +=  dc3 * phi[i] * fe_values_c.JxW (q);

            }
          }


        cell->get_dof_indices (local_dof_indices);
        constraints_c.distribute_local_to_global (cell_mass_matrix,
                                                  local_dof_indices,
                                                  mass_matrix);
        constraints_c.distribute_local_to_global (cell_rhs_c1,
                                                  local_dof_indices,
                                                  system_rhs_c1);
        constraints_c.distribute_local_to_global (cell_rhs_c2,
                                                  local_dof_indices,
                                                  system_rhs_c2);
        constraints_c.distribute_local_to_global (cell_rhs_c3,
                                                  local_dof_indices,
                                                  system_rhs_c3);

     }

    mass_matrix.compress (VectorOperation::add);
    system_rhs_c1.compress (VectorOperation::add);
    system_rhs_c2.compress (VectorOperation::add);
    system_rhs_c3.compress (VectorOperation::add);



  }

///////////////////////////////////////////////////////////////////////
// Setup quadrature point history
  template <int dim>
  void Solid<dim>::setup_qph()
  {

    {
    unsigned int our_cells = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned())
        ++our_cells;
    triangulation.clear_user_data();
    {
      std::vector<PointHistory<dim> > tmp;
      tmp.swap (quadrature_point_history);
    }
    quadrature_point_history.resize (our_cells * n_q_points);

      unsigned int history_index = 0;
      for (typename Triangulation<dim>::active_cell_iterator
              cell = triangulation.begin_active();
              cell != triangulation.end(); ++cell)
       if (cell->is_locally_owned())
        {
          cell->set_user_pointer(&quadrature_point_history[history_index]);
          history_index += n_q_points;
        }

      Assert(history_index == quadrature_point_history.size(),
             ExcInternalError());
    }

    for (typename Triangulation<dim>::active_cell_iterator
            cell = triangulation.begin_active();
            cell != triangulation.end(); ++cell)
     if (cell->is_locally_owned())
      {
        PointHistory<dim> *lqph =
          reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

        Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
        Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point].setup_lqp(parameters);
      }
  }
/////////////////////////////////////////////////////////
// Updates the data at all quadrature points over a loop run
  template <int dim>
  void Solid<dim>::update_qph_incremental()

  {
    FEValues<dim> fe_values (fe, qf_cell,
                                 update_values | update_gradients| update_quadrature_points);
    FEValues<dim> fe_values_c (fe_c, qf_cell,
                                     update_values | update_gradients| update_hessians);

    std::vector<Tensor<2, dim> > solution_grads_values (qf_cell.size());
    std::vector<double> solution_c1_values (qf_cell.size());
    std::vector<double> solution_c2_values (qf_cell.size());
    std::vector<double> solution_c3_values (qf_cell.size());

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    typename DoFHandler<dim>::active_cell_iterator
    cell_c = dof_handler_c.begin_active();
    for (; cell!=endc; ++cell, ++cell_c)
        if (cell->is_locally_owned())
        {
            PointHistory<dim> *lqph =
              reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

            Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
            Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());

            Assert(solution_grads_values.size() == n_q_points,
                   ExcInternalError());
            Assert(solution_c1_values.size() == n_q_points,
                   ExcInternalError());

            fe_values.reinit(cell);
            fe_values_c.reinit (cell_c);

            fe_values[u_fe].get_function_gradients(solution,  solution_grads_values);
            fe_values_c.get_function_values(solution_c1,   solution_c1_values);
            fe_values_c.get_function_values(solution_c2,   solution_c2_values);
            fe_values_c.get_function_values(solution_c3,   solution_c3_values);

           for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
           {

               lqph[q_point].update_values(solution_grads_values[q_point],
                                           solution_c1_values[q_point],
                                           solution_c2_values[q_point],
                                           solution_c3_values[q_point],
                                           parameters.delta_t,
                                           parameters.L,
                                                  fe_values.quadrature_point(q_point));
           }
       }

  }

///////////////////////////////////////////////////////////
// Output results: displacement, phase volume fraction c, and stresses
  template <int dim>
  void Solid<dim>::output_results() const
  {
    DataOut<dim> data_out;

// Output displacement and c
    std::vector<std::string> displacement_names;
         switch (dim)
           {
           case 1:
             displacement_names.push_back ("displacement");
               break;
           case 2:
               displacement_names.push_back ("x_displacement");
               displacement_names.push_back ("y_displacement");
               break;
           case 3:
               displacement_names.push_back ("x_displacement");
               displacement_names.push_back ("y_displacement");
               displacement_names.push_back ("z_displacement");
               break;
          default:
               Assert (false, ExcNotImplemented());
          }

     data_out.add_data_vector (dof_handler, solution, displacement_names);
     data_out.add_data_vector (dof_handler_c, solution_c0, "c0");
     data_out.add_data_vector (dof_handler_c, solution_c1, "c1");
     data_out.add_data_vector (dof_handler_c, solution_c2, "c2");
     data_out.add_data_vector (dof_handler_c, solution_c3, "c3");

/////////////////////////
// Output norm of stress field
    Vector<double> norm_of_stress (triangulation.n_active_cells());


     {
       typename Triangulation<dim>::active_cell_iterator
       cell = triangulation.begin_active(),
       endc = triangulation.end();
       for (; cell!=endc; ++cell)
           if (cell->is_locally_owned())
           {
             SymmetricTensor<2,dim> accumulated_stress;
             for (unsigned int q=0; q<qf_cell.size(); ++q)
               accumulated_stress +=
                 reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].get_tau();
             norm_of_stress(cell->active_cell_index())
               = (accumulated_stress /
                  qf_cell.size()).norm();
           }
           else
            norm_of_stress(cell->active_cell_index()) = -1e+20;
      }
data_out.add_data_vector (norm_of_stress, "norm_of_stress");

///////////////////////////////////////////////
//Output stress componenets
std::vector< std::vector< Vector<double> > >
     history_field_stress (dim, std::vector< Vector<double> >(dim)),
     local_history_values_at_qpoints_stress (dim, std::vector< Vector<double> >(dim)),
     local_history_fe_values_stress (dim, std::vector< Vector<double> >(dim));

   for (unsigned int i=0; i<dim; i++)
     for (unsigned int j=0; j<dim; j++)
     {
       history_field_stress[i][j].reinit(history_dof_handler.n_dofs());
       local_history_values_at_qpoints_stress[i][j].reinit(qf_cell.size());
       local_history_fe_values_stress[i][j].reinit(history_fe.dofs_per_cell);
     }

   Vector<double> history_field_drivingforce_c1,
                  history_field_drivingforce_c2,
                  history_field_drivingforce_c3,
                  local_history_values_at_qpoints_drivingforce_c1,
                  local_history_values_at_qpoints_drivingforce_c2,
                  local_history_values_at_qpoints_drivingforce_c3,
                  local_history_fe_values_drivingforce_c1,
                  local_history_fe_values_drivingforce_c2,
                  local_history_fe_values_drivingforce_c3;

   history_field_drivingforce_c1.reinit(history_dof_handler.n_dofs());
   history_field_drivingforce_c2.reinit(history_dof_handler.n_dofs());
   history_field_drivingforce_c3.reinit(history_dof_handler.n_dofs());
   local_history_values_at_qpoints_drivingforce_c1.reinit(qf_cell.size());
   local_history_values_at_qpoints_drivingforce_c2.reinit(qf_cell.size());
   local_history_values_at_qpoints_drivingforce_c3.reinit(qf_cell.size());
   local_history_fe_values_drivingforce_c1.reinit(history_fe.dofs_per_cell);
   local_history_fe_values_drivingforce_c2.reinit(history_fe.dofs_per_cell);
   local_history_fe_values_drivingforce_c3.reinit(history_fe.dofs_per_cell);


   FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
                                               qf_cell.size());
   FETools::compute_projection_from_quadrature_points_matrix
             (history_fe,
                 qf_cell, qf_cell,
              qpoint_to_dof_matrix);

   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end(),
   dg_cell = history_dof_handler.begin_active();
   for (; cell!=endc; ++cell, ++dg_cell)
     if (cell->is_locally_owned())
     {
       PointHistory<dim> *lqph
              = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
       Assert (lqph >=
                   &quadrature_point_history.front(),
                   ExcInternalError());
       Assert (lqph <
                   &quadrature_point_history.back(),
                   ExcInternalError());
       for (unsigned int i=0; i<dim; i++)
         for (unsigned int j=0; j<dim; j++)
         {
           for (unsigned int q=0; q<qf_cell.size(); ++q)
             {

              local_history_values_at_qpoints_stress[i][j](q) = (lqph[q].get_tau()[i][j])/(lqph[q].get_det_F());
              qpoint_to_dof_matrix.vmult (local_history_fe_values_stress[i][j],
                                          local_history_values_at_qpoints_stress[i][j]);
              dg_cell->set_dof_values (local_history_fe_values_stress[i][j],
                                       history_field_stress[i][j]);

              local_history_values_at_qpoints_drivingforce_c1(q) = lqph[q].get_update_c1();
              qpoint_to_dof_matrix.vmult (local_history_fe_values_drivingforce_c1,
                                          local_history_values_at_qpoints_drivingforce_c1);
              dg_cell->set_dof_values (local_history_fe_values_drivingforce_c1,
                                       history_field_drivingforce_c1);

              local_history_values_at_qpoints_drivingforce_c2(q) = lqph[q].get_update_c2();
              qpoint_to_dof_matrix.vmult (local_history_fe_values_drivingforce_c2,
                                          local_history_values_at_qpoints_drivingforce_c2);
              dg_cell->set_dof_values (local_history_fe_values_drivingforce_c2,
                                       history_field_drivingforce_c2);

              local_history_values_at_qpoints_drivingforce_c3(q) = lqph[q].get_update_c3();
              qpoint_to_dof_matrix.vmult (local_history_fe_values_drivingforce_c3,
                                          local_history_values_at_qpoints_drivingforce_c3);
              dg_cell->set_dof_values (local_history_fe_values_drivingforce_c3,
                                       history_field_drivingforce_c3);
         }
        }
     }

   std::vector<DataComponentInterpretation::DataComponentInterpretation>
                 data_component_interpretation2(1, DataComponentInterpretation::component_is_scalar);


   data_out.add_data_vector(history_dof_handler, history_field_stress[0][0], "sigma_11",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[1][1], "sigma_22",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[2][2], "sigma_33",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[0][1], "sigma_12",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[0][2], "sigma_13",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[1][2], "sigma_23",
                                data_component_interpretation2);


   data_out.add_data_vector(history_dof_handler, history_field_drivingforce_c1, "dc1",
                                   data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_drivingforce_c2, "dc2",
                                      data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_drivingforce_c3, "dc3",
                                      data_component_interpretation2);
//////////////////////////
// writing output files
    MappingQEulerian<dim, vectorType > q_mapping(degree, dof_handler, solution);

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches(q_mapping, degree);

    const unsigned int cycle = time.get_timestep();

    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
         filenames.push_back ("solution-" +
                               Utilities::int_to_string (cycle, 2) +
                               "." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");
         std::ofstream master_output (("solution-" +
                                      Utilities::int_to_string (cycle, 2) +
                                      ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }

   }

//////////////////////////////////
//Output average or global data such as stress and strain on the external boundary to be used for stress-strain curves
template <int dim>
void Solid<dim>::output_resultant_stress()
    {

            FEFaceValues<dim> fe_face_values (fe, qf_face,
                                                 update_values         | update_quadrature_points  |
                                                 update_normal_vectors | update_JxW_values);
               double resultant_force = 0.0;
               double resultant_pseudo_force = 0.0;
               double current_face_area = 0.0;
               double referrence_face_area = 0.0;
               double resultant_E = 0.0;
               typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end();
               for (; cell != endc; ++cell)
                 if (cell->is_locally_owned())
               {

                   PointHistory<dim> *lqph =
                                             reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

                    for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)

                      if (cell->face(face)->at_boundary()
                         &&
                         (cell->face(face)->boundary_id() == 5))
                       {
                         fe_face_values.reinit (cell, face);
                         for (unsigned int q_point=0; q_point<n_q_points_f; ++q_point)
                           {
                             const Tensor<2, dim> F_inv_tr = lqph[q_point].get_F_inv_tr();
                             const Tensor<2, dim> F_inv = lqph[q_point].get_F_inv();
                             const Tensor<2, dim> tau      = lqph[q_point].get_tau();
                             const Tensor<2, dim> E      = lqph[q_point].get_E();
                             const double J = lqph[q_point].get_det_F();

                             const Tensor<1, dim> element_force
                                     = (tau*F_inv_tr)* fe_face_values.normal_vector(q_point);
                             const Tensor<1, dim> element_pseudo_force
                                     =F_inv*element_force;

                             const double current_element_area = J * ((F_inv_tr*fe_face_values.normal_vector(q_point)).norm());

                             resultant_force += element_force [2]*fe_face_values.JxW(q_point);
                             resultant_pseudo_force += element_pseudo_force [2]*fe_face_values.JxW(q_point);
                               current_face_area += current_element_area* fe_face_values.JxW(q_point);
                               referrence_face_area += fe_face_values.JxW(q_point);
                               resultant_E += E[2][2]*fe_face_values.JxW(q_point);
                           }
                       }
               }


                resultant_cauchy_stress[time.get_timestep()]=-1*Utilities::MPI::sum(resultant_force, mpi_communicator )/
                                                        Utilities::MPI::sum(current_face_area, mpi_communicator );
                resultant_first_piola_stress[time.get_timestep()]=-1*Utilities::MPI::sum(resultant_force, mpi_communicator )/
                                                       Utilities::MPI::sum(referrence_face_area, mpi_communicator );
                resultant_second_piola_stress[time.get_timestep()]=-1*Utilities::MPI::sum(resultant_pseudo_force, mpi_communicator )/
                                                       Utilities::MPI::sum(referrence_face_area, mpi_communicator );
                resultant_lagrangian_strain[time.get_timestep()]=-1*Utilities::MPI::sum(resultant_E, mpi_communicator )/
                                                            Utilities::MPI::sum(referrence_face_area, mpi_communicator );

            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
              {

                std::ofstream myfile_1;
                std::ofstream myfile_2;
                std::ofstream myfile_3;
                std::ofstream myfile_4;
                std::ofstream myfile_5;
                myfile_1.open ("resultant_cauchy_stress.txt");
                myfile_2.open ("resultant_first_piola_stress.txt");
                myfile_3.open ("resultant_second_piola_stress.txt");
                myfile_4.open ("resultant_lagrangian_strain.txt");
                myfile_5.open ("order_parametre.txt");
                for(unsigned int n=0; n<time.get_timestep(); n++)
                {
                  myfile_1<< resultant_cauchy_stress[n]<<std::endl;
                  myfile_2<< resultant_first_piola_stress[n]<<std::endl;
                  myfile_3<< resultant_second_piola_stress[n]<<std::endl;
                   myfile_4<< resultant_lagrangian_strain[n]<<std::endl;
                  myfile_5<< order_parameter[n]<<std::endl;
                }
                  myfile_1.close();
                  myfile_2.close();
                  myfile_3.close();
                  myfile_4.close();
                  myfile_5.close();

           }
        }
//////////////////////////////
// Solve mechanics problem
   template <int dim>
   unsigned int
   Solid<dim>::solve ()
     {

       vectorType
       completely_distributed_solution (locally_owned_dofs, mpi_communicator);

       SolverControl solver_control (dof_handler.n_dofs(), 1e-6*system_rhs.l2_norm());

       TrilinosWrappers::SolverCG solver(solver_control);


       TrilinosWrappers::PreconditionAMG preconditioner;
       //TrilinosWrappers::PreconditionSSOR preconditioner;
       preconditioner.initialize(tangent_matrix);
       solver.solve (tangent_matrix, completely_distributed_solution, system_rhs,
                           preconditioner);

 //      TrilinosWrappers::SolverDirect::AdditionalData data;
 //      data.solver_type= "Amesos_Superludist";
 //      TrilinosWrappers::SolverDirect solver (solver_control, data);
 //      solver.solve (tangent_matrix, completely_distributed_solution, system_rhs);

       constraints.distribute (completely_distributed_solution);

       solution_update = completely_distributed_solution;

       return solver_control.last_step();
     }

///////////////////////////////////
//Solve phase-field kinetic equation
   template <int dim>
     void
     Solid<dim>::solve_c ( )
      {
        assemble_system_c ();
         vectorType   solution_update_c1 (locally_owned_dofs_c, mpi_communicator);
         vectorType   solution_update_c2 (locally_owned_dofs_c, mpi_communicator);
         vectorType   solution_update_c3 (locally_owned_dofs_c, mpi_communicator);

         vectorType   temp_solution_c1 (locally_owned_dofs_c, mpi_communicator);
         vectorType   temp_solution_c2 (locally_owned_dofs_c, mpi_communicator);
         vectorType   temp_solution_c3 (locally_owned_dofs_c, mpi_communicator);

         temp_solution_c1= solution_c1;
         temp_solution_c2= solution_c2;
         temp_solution_c3= solution_c3;


         const double success_tol_c1 = (system_rhs_c1.l2_norm()==0)? 1.e-6 : 1e-6*system_rhs_c1.l2_norm();
         const double success_tol_c2 = (system_rhs_c2.l2_norm()==0)? 1.e-6 : 1e-6*system_rhs_c2.l2_norm();
         const double success_tol_c3 = (system_rhs_c3.l2_norm()==0)? 1.e-6 : 1e-6*system_rhs_c3.l2_norm();

         SolverControl solver_control_c1 (dof_handler_c.n_dofs(), success_tol_c1);
         SolverControl solver_control_c2 (dof_handler_c.n_dofs(), success_tol_c2);
         SolverControl solver_control_c3 (dof_handler_c.n_dofs(), success_tol_c3);

         TrilinosWrappers::SolverCG solver_c1 (solver_control_c1);
         TrilinosWrappers::SolverCG solver_c2 (solver_control_c2);
         TrilinosWrappers::SolverCG solver_c3 (solver_control_c3);

         TrilinosWrappers::PreconditionAMG preconditioner;
         preconditioner.initialize(mass_matrix);

         solver_c1.solve (mass_matrix, solution_update_c1, system_rhs_c1, preconditioner);
         solver_c2.solve (mass_matrix, solution_update_c2, system_rhs_c2, preconditioner);
         solver_c3.solve (mass_matrix, solution_update_c3, system_rhs_c3, preconditioner);

         constraints_c.distribute (solution_update_c1);
         constraints_c.distribute (solution_update_c2);
         constraints_c.distribute (solution_update_c3);

         temp_solution_c1 += solution_update_c1;
         temp_solution_c2 += solution_update_c2;
         temp_solution_c3 += solution_update_c3;

         IndexSet::ElementIterator it=locally_owned_dofs_c.begin()  ;
         unsigned int n_dofs_per_core = locally_owned_dofs_c.n_elements();

         for(unsigned int i = 0; i < n_dofs_per_core; i++)
            {
              if (temp_solution_c1(*it)<  0)
                  temp_solution_c1(*it) = 0;
              if (temp_solution_c1(*it)>  1)
                  temp_solution_c1(*it) = 1;

              if (temp_solution_c2(*it)<  0)
                  temp_solution_c2(*it) = 0;
              if (temp_solution_c2(*it)>  1)
                  temp_solution_c2(*it) = 1;

              if (temp_solution_c3(*it)<  0)
                  temp_solution_c3(*it) = 0;
              if (temp_solution_c3(*it)>  1)
                  temp_solution_c3(*it) = 1;


              it++;


            }

         solution_c1= temp_solution_c1;
         solution_c2= temp_solution_c2;
         solution_c3= temp_solution_c3;

         update_qph_incremental();

         vectorType   temp_solution_c0 (locally_owned_dofs_c, mpi_communicator);
         temp_solution_c0=1;
         temp_solution_c0 -= (temp_solution_c1+temp_solution_c2+temp_solution_c3);
         temp_solution_c0.compress(VectorOperation::add);
         solution_c0= temp_solution_c0;

         pcout << "Solving for Volume fractions: \n"
               << solver_control_c1.last_step()<<"+"
               <<solver_control_c1.last_step()<<"+"
               <<solver_control_c1.last_step()
               << "   CG Solver iterations for C1, C2 and C3."
                 << std::endl;


         }


   /////////////////////////////
// Solve Newton-Raphson iterative alqorithm to solve nonlinear mechanical problem
      template <int dim>
      void Solid<dim>::solve_nonlinear_timestep()
       {
         double initial_rhs_norm = 0.;
         unsigned int newton_iteration = 0;
         unsigned int n_iterations=0;
         vectorType  temp_solution_update(locally_owned_dofs, mpi_communicator);
         vectorType  tmp(locally_owned_dofs, mpi_communicator);
         tmp = solution;
           for (; newton_iteration < 100;   ++newton_iteration)
             {
               make_constraints(newton_iteration);
               assemble_system ();


                 if (newton_iteration == 0){
                initial_rhs_norm = system_rhs.l2_norm();
                pcout << " Solving for Displacement:   " << std::endl;
              }
                 pcout<<"   rhs_norm : "<<system_rhs.l2_norm();

              // tangent_matrix.print(pcout) ;
              // system_rhs.print(pcout);


                n_iterations = solve ();
                pcout << "    Number of CG iterations: " << n_iterations<< std::endl;

                 temp_solution_update = solution_update;


                 tmp += temp_solution_update;
                 solution = tmp;

                 update_qph_incremental();




                if (newton_iteration > 0 && system_rhs.l2_norm() <= 1e-4 * initial_rhs_norm)
                 {
                  pcout << "CONVERGED! " << std::endl;
                  break;
                 }
               AssertThrow (newton_iteration < 99,
               ExcMessage("No convergence in nonlinear solver!"));

            }
       }


  //////////////////////////////////////////////////
      template <int dim>
      void Solid<dim>::run()
      {
        make_grid(); // generates the geometry and mesh
        system_setup(); // sets up the system matrices and RHS

        // Applying initial condition for volume fraction c
        vectorType  tmp_solution_c1(locally_owned_dofs_c, mpi_communicator);
        vectorType  tmp_solution_c2(locally_owned_dofs_c, mpi_communicator);
        vectorType  tmp_solution_c3(locally_owned_dofs_c, mpi_communicator);
        VectorTools::interpolate(dof_handler_c, InitialValues<dim>(1,0), tmp_solution_c1); //initial c
        VectorTools::interpolate(dof_handler_c, InitialValues<dim>(2,0), tmp_solution_c2); //initial c
        VectorTools::interpolate(dof_handler_c, InitialValues<dim>(3,0), tmp_solution_c3); //initial c
        solution_c1= tmp_solution_c1;
        solution_c2= tmp_solution_c2;
        solution_c3= tmp_solution_c3;

        update_qph_incremental();

        solve_nonlinear_timestep();
        output_results();
        output_resultant_stress();

        time.increment();

     // computed actual time integration to update displacement and c
        while (time.current() <= time.end() )
          {

            pcout << std::endl
                  << "Time step #" << time.get_timestep() << "; "
                  << "advancing to t = " << time.current() << "."
                  << std::endl;


//        if (/*time.get_timestep()>3 &&*/ load_step<12 )
//        load_step += 2;

         solve_c();

         solve_nonlinear_timestep();

         if(time.get_timestep()%50 == 0)
          {
           output_results();
          }
         output_resultant_stress();

         time.increment();

          }
       }


}
/////////////////////////////////////////////////////////
  int main (int argc, char *argv[])
  {
    try
      {
        using namespace dealii;
        using namespace PhaseField;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        Solid<3> solid_3d("parameters.prm");
        solid_3d.run();
      }
    catch (std::exception &exc)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl << exc.what()
                  << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
      }
    catch (...)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
      }

    return 0;
  }

