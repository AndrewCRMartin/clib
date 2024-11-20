/* #define DEBUG 1  */
#define TEST  1
/*
  Multidimensional minimization using the Nealder & Mead
  downhill simplex method.

  Given a function ('funct()') that takes, and evaluates, an
  N-dimensional vector, we create an (N+1 x N) dimensional
  simplex ('simplex') as a set of starting values around the
  initial vector that needs to be minimized.
  
  The function also takes a vector 'evaluations' of length N+1
  pre-initialized to the values of funct() evaluated at the N+1
  vertices (rows) of the simplex; and 'tol' the fractional
  convergence tolerance to be achieved in the function value.
  On output, simplex and evaluations will have been reset to the
  N+1 new points all within tol of a minimum function value, and
  it returns the number of iterations taken.
*/

#include <stdio.h>
#include <math.h>
#include "bioplib/array.h"
#include "bioplib/MathType.h"

#define ERR_STUCK            -1
#define ERR_MAXITER          -2
#define ERR_NOMEM            -3
#ifndef FREE
   #define FREE(x) if(x != NULL) {free(x); x=NULL;}
#endif

#define EXTRAPOLATION_FACTOR 1.0        /* Extrapolation                */
#define CONTRACTION_FACTOR   0.5        /* Contraction                  */
#define ADDEXTRAP_FACTOR     2.0        /* Additional extrapolation     */
#define TINY                 0.0000001  /* Residual converged           */
#define N_RESIDUALS_CHECK    10         /* Number of residuals to check */


/************************************************************************/
static void ReplaceVertex(REAL **simplex, int ndim, int worstVertex,
                          REAL *newVertex)
{
   int i;
   for(i=1; i<=ndim; i++)
   {
      simplex[worstVertex][i] = newVertex[i];
   }
}

/************************************************************************/
static int ContractSimplexAroundBest(REAL **simplex, int ndim,
                                     int bestVertex, REAL *evaluations,
                                     REAL (*funct)(REAL *x, int nd))
{
   int i, j,
       nVertices = ndim+1;
   REAL *midpoint = NULL;
   if((midpoint = (REAL *)malloc((ndim+1)*sizeof(REAL)))==NULL)
   {
      return(ERR_NOMEM);
   }
   
   for(i=1; i<=nVertices; i++)
   {
      if(i != bestVertex)
      {
         for(j=1; j<=ndim; j++)
         {
            midpoint[j] = 0.5*(simplex[i][j] + simplex[bestVertex][j]);
            simplex[i][j] = midpoint[j];
         }
         evaluations[i] = (*funct)(midpoint,ndim);
      }
   }
   FREE(midpoint);
   return(0);
}

/************************************************************************/
static REAL ContractSimplex(REAL **simplex, int ndim, int worstVertex,
                            REAL *extraVertex, REAL *centreWithoutWorst,
                            REAL (*funct)(REAL *x, int nd))
{
   int j;

   /* Contract the simplex along one dimension                          */
   for(j=1; j<=ndim; j++)
   {
      extraVertex[j] =
         (CONTRACTION_FACTOR * simplex[worstVertex][j]) +
         ((1.0-CONTRACTION_FACTOR) * centreWithoutWorst[j]);
   }
   return((*funct)(extraVertex,ndim));
}

/************************************************************************/
static REAL AdditionalExtrapolate(REAL **simplex,
                                  REAL *centreWithoutWorst,
                                  int ndim, REAL *newVertex,
                                  REAL *extraVertex,
                                  REAL (*funct)(REAL *x, int nd))
{
   int j;

   /* Try an additional extrapolation by ADDEXTRAP_FACTOR               */
   for(j=1; j<=ndim; j++)   /** WAS < ndim and worked! **/
   {
      extraVertex[j] =
         (ADDEXTRAP_FACTOR * newVertex[j]) +
         ((1.0 - ADDEXTRAP_FACTOR) * centreWithoutWorst[j]);
   }
   
   return((*funct)(extraVertex,ndim));
}


/************************************************************************/
/* Extrapolate by a factor EXTRAPOLATION_FACTOR through the face
   i.e. reflect the simplex from the worst vertex and evaluate
*/
static REAL Extrapolate(REAL **simplex, int worstVertex,
                        REAL *centreWithoutWorst, int ndim,
                        REAL *newVertex,
                        REAL (*funct)(REAL *x, int nd))
{
   int i;

   for(i=1; i<=ndim; i++)
   {
      newVertex[i] =
         ((1.0 + EXTRAPOLATION_FACTOR)*centreWithoutWorst[i]) -
         (EXTRAPOLATION_FACTOR * simplex[worstVertex][i]);
   }

   return((*funct)(newVertex,ndim));
}

/************************************************************************/
static int checkVal(REAL *values, int nValues, REAL newVal)
{
   static int n    = 0,
              full = 0;

   values[n++] = newVal;
   if(n >= nValues)
   {
      n    = 0;
      full = 1;
   }

   if(full)
   {
      int i;
      for(i=1; i<nValues; i++)
      {
         if((values[i] >= (values[0] - TINY)) &&
            (values[i] <= (values[0] + TINY)))
         {
            return(1);
         }
      }
   }
   return(0);
}

/************************************************************************/
static void FindBestAndWorstVertices(REAL *evaluations, int nVertices,
                                     int *worstVertex,
                                     int *nextWorstVertex,
                                     int *bestVertex)
{
   int i;

   /* Find which vertex has the worst score, the next highest score, and
      the best score.
   */
   if(evaluations[1] > evaluations[2])
   {
      *worstVertex     = 1;
      *nextWorstVertex = 2;
   }
   else
   {
      *worstVertex     = 2;
      *nextWorstVertex = 1;
   }
   
   for(i=1; i<=nVertices; i++)
   {
      if(evaluations[i] < evaluations[*bestVertex])
         *bestVertex = i;
      
      if(evaluations[i] > evaluations[*worstVertex])
      {
         *nextWorstVertex = *worstVertex;
         *worstVertex     = i;
      }
      else if(evaluations[i] > evaluations[*nextWorstVertex])
      {
         if(i != *worstVertex)
            *nextWorstVertex = i;
      }
   }
}


/************************************************************************/
int NMDownhillSimplex(REAL **simplex, REAL *evaluations, int ndim,
                      REAL tol, int maxIter,
                      REAL (*funct)(REAL *x, int nd))
{
   REAL *newVertex          = (REAL *)malloc((ndim+1)*sizeof(REAL));
   REAL *extraVertex        = (REAL *)malloc((ndim+1)*sizeof(REAL));
   REAL *centreWithoutWorst = (REAL *)malloc((ndim+1)*sizeof(REAL));
   REAL nVertices   = ndim+1;
   int  iter   = 0,
        retval = 0;

   REAL residuals[N_RESIDUALS_CHECK];

   if((newVertex = (REAL *)malloc((ndim+1)*sizeof(REAL)))==NULL)
      retval=ERR_NOMEM;
      
   if((extraVertex = (REAL *)malloc((ndim+1)*sizeof(REAL)))==NULL)
      retval=ERR_NOMEM;
   
   if((centreWithoutWorst = (REAL *)malloc((ndim+1)*sizeof(REAL)))==NULL)
      retval=ERR_NOMEM;

   if(retval == ERR_NOMEM)
   {
      FREE(newVertex);
      FREE(extraVertex);
      FREE(centreWithoutWorst);

      return(retval);
   }
   
   while(1)
   {
      int  bestVertex = 1,
           worstVertex, nextWorstVertex,
           i, j;
      REAL residual, evalNewVertex, evalExtraVertex;

      FindBestAndWorstVertices(evaluations, ndim+1, &worstVertex,
                               &nextWorstVertex, &bestVertex);

      /* Calculate a residual as the fractional range:
            2*|worst-best|/(|worst|+|best|)
         and return if low enough
      */
      residual =
         2.0 * fabs(evaluations[worstVertex]-evaluations[bestVertex]) /
         (fabs(evaluations[worstVertex]) + fabs(evaluations[bestVertex]));
#ifdef DEBUG
      printf("Residual: %.9f\n", residual);
#endif
      if(residual < tol)
      {
         retval = iter;
      }
      else if(checkVal(residuals, N_RESIDUALS_CHECK, residual))
      {
         retval = ERR_STUCK;
      }
      else if(iter == maxIter)
      {
         retval = ERR_MAXITER;
      }
      
      if(retval != 0)
      {
         FREE(newVertex);
         FREE(extraVertex);
         FREE(centreWithoutWorst);

         return(retval);
      }

      iter++;

      /* Reset the array of averages                                    */
      for(j=1; j<=ndim; j++)
         centreWithoutWorst[j]=0.0;

      /* Begin a new iteration.  Compute the vector average of all
         points except the highest i.e. the centre of the "face" of the
         simplex across from the high point.  We will subsequently
         explore along the ray from the high point through that centre.
      */
      for(i=1; i<=nVertices; i++)
      {
         if(i != worstVertex)
         {
            for(j=1; j<=ndim; j++)
            {
               centreWithoutWorst[j] += simplex[i][j];
            }
         }
      }
      /* And calculate the means */
      for(j=1; j<=ndim; j++)
      {
         centreWithoutWorst[j] /= ndim;
      }

      /* Extrapolate the worst point                                    */
      evalNewVertex = Extrapolate(simplex, worstVertex,
                                  centreWithoutWorst, ndim, newVertex,
                                  funct);
      if(evalNewVertex <= evaluations[bestVertex])
      {
         /* If it gives a result better than the best point so far, then
            try and additional extrapolation
         */
         evalExtraVertex = AdditionalExtrapolate(simplex,
                                                 centreWithoutWorst, ndim,
                                                 newVertex, extraVertex,
                                                 funct);
         if(evalExtraVertex < evaluations[bestVertex])
         {
            /* The additional extrapolation gave an even better value,
               so replace the current worst (high) point
            */
            ReplaceVertex(simplex, ndim, worstVertex, extraVertex);
            evaluations[worstVertex] = evalExtraVertex;
         }
         else 
         {
            /* The additional extrapolation failed, so just use the
               first extrapolation
            */
            ReplaceVertex(simplex, ndim, worstVertex, newVertex);
            evaluations[worstVertex] = evalNewVertex;
         }
      }
      else if(evalNewVertex >= evaluations[nextWorstVertex])
      {
         /* The extrapolation wasn't better than the best point, but it's
            better than the second worst, so test if it's better than the
            worst was
         */
         if(evalNewVertex < evaluations[worstVertex])
         {
            /* It's better than the worst vertex so replace it          */
            ReplaceVertex(simplex, ndim, worstVertex, newVertex);
            evaluations[worstVertex] = evalNewVertex;
         }

         /* Look for an intermediate better point by contracting the
            simplex
         */
         evalExtraVertex = ContractSimplex(simplex, ndim, worstVertex,
                                           extraVertex,
                                           centreWithoutWorst, funct);
         if(evalExtraVertex < evaluations[worstVertex])
         {
            /* The contraction has given an improvement, so accept it   */
            for(j=1; j<=ndim; j++)
            {
               simplex[worstVertex][j] = extraVertex[j];
            }
            evaluations[worstVertex] = evalExtraVertex;
         }
         else
         {
            /* The contraction didn't give an improvement, so contract
               the simplex around the lowest (best) point.
            */
            retval = ContractSimplexAroundBest(simplex, ndim, bestVertex,
                                               evaluations, funct);
            if(retval != 0)
               return(retval);
         }
      }
      else 
      {
         /* The original extrapolation didn't improve on the worst or
            second-worst vertex, so just replace the worst vertex.
         */
         ReplaceVertex(simplex, ndim, worstVertex, newVertex);
         evaluations[worstVertex] = evalNewVertex;
      }
   }
   return(iter);
}


REAL FindSimplexCentroid(REAL **simplex, int ndim, REAL *centroid,
                         REAL (*funct)(REAL *x, int nd))
{
   int nVertices = ndim+1,
       i, j;
   
   for(i=1; i<=ndim; i++)          /* For each dimension                */
   {
      centroid[i] = 0.0;
      for(j=1; j<=nVertices; j++)  /* For each vertex                   */
      {
         centroid[i] += simplex[j][i];
      }
      centroid[i] /= nVertices;
   }

   return((*funct)(centroid,ndim));
}


/************************************************************************
 **                             TEST CODE                              **
 ************************************************************************/

#ifdef TEST
/* The function to be evaluated                                         */
REAL energy(REAL *inp, int ndim)
{
   int i;
   REAL e = 0.0;
   for(i=1; i<=ndim; i++)
   {
      e += (REAL)i - inp[i];
   }
   return(e);
}

/* Initialize the simplex. In reality, this would take the starting
   point as an input. This would be copied to the vertices and then the
   lambda modifications made
*/
void InitializeSimplex(REAL **simplex, int ndim, REAL lambda)
{
   int ivec, idim;
   
   /* Initial guess - set all points to this                            */
   for(ivec=1; ivec<=ndim+1; ivec++)
   {
      for(idim=1; idim<=ndim; idim++)
      {
         simplex[ivec][idim] = idim;
      }
   }

   /* Update all, but first, dimensions of simplex                      */
   for(ivec=2; ivec<=ndim+1; ivec++)
   {
      idim = ivec-1;
      simplex[ivec][idim] += lambda;
   }
}

/* Initialize the evaluations array with the values for each vertex of
   the simplex
*/
void InitializeEvaluations(REAL *evaluations, REAL **simplex, int ndim)
{
   int i;
   for(i=1; i<=ndim+1; i++)
   {
      evaluations[i] = energy(simplex[i], ndim);
   }
}

/* For testing                                                          */
#define NDIM 20
/* For creating the simplex                                             */
#define LAMBDA  10000
#define TOL     0.1
#define MAXITER 10000


int main(int argc, char **argv)
{
   REAL **simplex,
        evaluations[NDIM+2],
        centroid[NDIM+1],
        evalAtCentroid,
        tol = TOL;
   int  iter, i, j;

   if((simplex = (REAL **)blArray2D(sizeof(REAL), NDIM+2, NDIM+1))==NULL)
   {
      fprintf(stderr,"  ** No memory for the simplex.\n");
      return(1);
   }

   InitializeSimplex(simplex, NDIM, LAMBDA);
   InitializeEvaluations(evaluations, simplex, NDIM);
   iter = NMDownhillSimplex(simplex, evaluations, NDIM, tol,
                            MAXITER, energy);
   if(iter < 0)
   {
      switch(iter)
      {
      case ERR_STUCK:
         fprintf(stderr,"  ** Simplex is not improving.\n");
         break;
      case ERR_NOMEM:
         fprintf(stderr,"  ** No memory.\n");
         break;
      case ERR_MAXITER:
         fprintf(stderr,"  ** Max iterations exceeded (MAXITER=%d).\n",
                 MAXITER);
         break;
      }
   }

   if(iter > 0)
      printf("Iterations: %d\n", iter);

   printf("Simplex Energies...\n");
   for(i=1; i<=NDIM+1; i++)
   {
      printf("Vtx %3d Coor: ", i);
      for(j=1; j<=NDIM; j++)
      {
         printf("%4.2g ", simplex[i][j]);
      }
      printf("E: %4.2g\n", evaluations[i]);
   }

   printf("\nSimplex Centroid...\n");
   evalAtCentroid = FindSimplexCentroid(simplex, NDIM, centroid, energy);
   for(i=1; i<=NDIM; i++)
   {
      printf("%4.2g ", centroid[i]);
   }
   printf("E: %4.2g\n", evalAtCentroid);
   
   
   if(iter<0)
      return(1);
   return(0);
}
#endif
