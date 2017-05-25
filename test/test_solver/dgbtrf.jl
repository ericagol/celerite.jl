#> \brief \b DGBTRF
#
#  =========== DOCUMENTATION ===========
#
# Online html documentation available at
#            http://www.netlib.org/lapack/explore-html/
#
#> \htmlonly
#> Download DGBTRF + dependencies
#> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgbtrf.f">
#> [TGZ]</a>
#> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgbtrf.f">
#> [ZIP]</a>
#> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgbtrf.f">
#> [TXT]</a>
#> \endhtmlonly
#
#  Definition:
#  ===========
#
#       SUBROUTINE DGBTRF( M, N, KL, KU, AB, LDAB, IPIV, INFO )
#
#       .. Scalar Arguments ..
#       INTEGER            INFO, KL, KU, LDAB, M, N
#       ..
#       .. Array Arguments ..
#       INTEGER            IPIV( * )
#       DOUBLE PRECISION   AB( LDAB, * )
#       ..
#
#
#> \par Purpose:
#  =============
#>
#> \verbatim
#>
#> DGBTRF computes an LU factorization of a real m-by-n band matrix A
#> using partial pivoting with row interchanges.
#>
#> This is the blocked version of the algorithm, calling Level 3 BLAS.
#> \endverbatim
#
#  Arguments:
#  ==========
#
#> \param[in] M
#> \verbatim
#>          M is INTEGER
#>          The number of rows of the matrix A.  M >= 0.
#> \endverbatim
#>
#> \param[in] N
#> \verbatim
#>          N is INTEGER
#>          The number of columns of the matrix A.  N >= 0.
#> \endverbatim
#>
#> \param[in] KL
#> \verbatim
#>          KL is INTEGER
#>          The number of subdiagonals within the band of A.  KL >= 0.
#> \endverbatim
#>
#> \param[in] KU
#> \verbatim
#>          KU is INTEGER
#>          The number of superdiagonals within the band of A.  KU >= 0.
#> \endverbatim
#>
#> \param[in,out] AB
#> \verbatim
#>          AB is DOUBLE PRECISION array, dimension (LDAB,N)
#>          On entry, the matrix A in band storage, in rows KL+1 to
#>          2*KL+KU+1; rows 1 to KL of the array need not be set.
#>          The j-th column of A is stored in the j-th column of the
#>          array AB as follows:
#>          AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl)
#>
#>          On exit, details of the factorization: U is stored as an
#>          upper triangular band matrix with KL+KU superdiagonals in
#>          rows 1 to KL+KU+1, and the multipliers used during the
#>          factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
#>          See below for further details.
#> \endverbatim
#>
#> \param[in] LDAB
#> \verbatim
#>          LDAB is INTEGER
#>          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.
#> \endverbatim
#>
#> \param[out] IPIV
#> \verbatim
#>          IPIV is INTEGER array, dimension (min(M,N))
#>          The pivot indices; for 1 <= i <= min(M,N), row i of the
#>          matrix was interchanged with row IPIV(i).
#> \endverbatim
#>
#> \param[out] INFO
#> \verbatim
#>          INFO is INTEGER
#>          = 0: successful exit
#>          < 0: if INFO = -i, the i-th argument had an illegal value
#>          > 0: if INFO = +i, U(i,i) is exactly zero. The factorization
#>               has been completed, but the factor U is exactly
#>               singular, and division by zero will occur if it is used
#>               to solve a system of equations.
#> \endverbatim
#
#  Authors:
#  ========
#
#> \author Univ. of Tennessee
#> \author Univ. of California Berkeley
#> \author Univ. of Colorado Denver
#> \author NAG Ltd.
#
#> \date December 2016
#
#> \ingroup doubleGBcomputational
#
#> \par Further Details:
#  =====================
#>
#> \verbatim
#>
#>  The band storage scheme is illustrated by the following example, when
#>  M = N = 6, KL = 2, KU = 1:
#>
#>  On entry:                       On exit:
#>
#>      *    *    *    +    +    +       *    *    *   u14  u25  u36
#>      *    *    +    +    +    +       *    *   u13  u24  u35  u46
#>      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
#>     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
#>     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
#>     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *
#>
#>  Array elements marked * are not used by the routine; elements marked
#>  + need not be set on entry, but are required by the routine to store
#>  elements of U because of fill-in resulting from the row interchanges.
#> \endverbatim
#>
#  =====================================================================
      function dgbtrf( M::Integer, N::Integer, KL::Integer, KU::Integer, AB::Array, LDAB::Integer, IPIV::Vector{Integer}, INFO::Integer)
#
#  -- LAPACK computational routine (version 3.7.0) --
#  -- LAPACK is a software package provided by Univ. of Tennessee,    --
#  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
#     December 2016
#
#     .. Scalar Arguments ..
#      INTEGER            INFO, KL, KU, LDAB, M, N
#     ..
#     .. Array Arguments ..
#      INTEGER            IPIV( * )
#      DOUBLE PRECISION   AB( ldab, * )
      @assert(size(ab,1) == LDAB)
#     ..
#
#  =====================================================================
#
#     .. Parameters ..

#    DOUBLE PRECISION   ONE, ZERO
#      parameter                ( one = 1.0d+0, zero = 0.0d+0 )
      INTEGER            NBMAX, LDWORK
#      parameter                ( nbmax = 64, ldwork = nbmax+1 )
      nbmax = 64
      ldwork = nbmax+1
#     ..
#     .. Local Scalars ..
#      INTEGER            I, I2, I3, II, IP, J, J2, J3, JB, JJ, JM, JP,
#     $                   ju, k2, km, kv, nb, nw
#      DOUBLE PRECISION   TEMP
#     ..
#     .. Local Arrays ..
#      DOUBLE PRECISION   WORK13( ldwork, nbmax ),
#     $                   work31( ldwork, nbmax )
      work13 = zeros(ldwork,nbmax)
      work31 = zeros(ldwork,nbmax)
#     ..
#     .. External Functions ..
      INTEGER            IDAMAX, ILAENV
      EXTERNAL           idamax, ilaenv
#     ..
#     .. External Subroutines ..
      EXTERNAL           dcopy, dgbtf2, dgemm, dger, dlaswp, dscal,
     $                   dswap, dtrsm, xerbla
#     ..
#     .. Intrinsic Functions ..
      INTRINSIC          max, min
#     ..
#     .. Executable Statements ..
#
#     KV is the number of superdiagonals in the factor U, allowing for
#     fill-in
#
      kv = ku + kl
#
#     Test the input parameters.
#
      info = 0
      IF( m.LT.0 ) THEN
         info = -1
      ELSE IF( n.LT.0 ) THEN
         info = -2
      ELSE IF( kl.LT.0 ) THEN
         info = -3
      ELSE IF( ku.LT.0 ) THEN
         info = -4
      ELSE IF( ldab.LT.kl+kv+1 ) THEN
         info = -6
      END IF
      IF( info.NE.0 ) THEN
         CALL xerbla( 'DGBTRF', -info )
         RETURN
      END IF
#
#     Quick return if possible
#
      IF( m.EQ.0 .OR. n.EQ.0 )
     $   RETURN
#
#     Determine the block size for this environment
#
      nb = ilaenv( 1, 'DGBTRF', ' ', m, n, kl, ku )
#
#     The block size must not exceed the limit set by the size of the
#     local arrays WORK13 and WORK31.
#
      nb = min( nb, nbmax )
#
      IF( nb.LE.1 .OR. nb.GT.kl ) THEN
#
#        Use unblocked code
#
         CALL dgbtf2( m, n, kl, ku, ab, ldab, ipiv, info )
      ELSE
#
#        Use blocked code
#
#        Zero the superdiagonal elements of the work array WORK13
#
         DO 20 j = 1, nb
            DO 10 i = 1, j - 1
               work13( i, j ) = zero
   10       CONTINUE
   20    CONTINUE
#
#        Zero the subdiagonal elements of the work array WORK31
#
         DO 40 j = 1, nb
            DO 30 i = j + 1, nb
               work31( i, j ) = zero
   30       CONTINUE
   40    CONTINUE
252 *
#        Gaussian elimination with partial pivoting
#
#        Set fill-in elements in columns KU+2 to KV to zero
#
         DO 60 j = ku + 2, min( kv, n )
            DO 50 i = kv - j + 2, kl
               ab( i, j ) = zero
   50       CONTINUE
   60    CONTINUE
#
#        JU is the index of the last column affected by the current
#        stage of the factorization
#
         ju = 1
#
         DO 180 j = 1, min( m, n ), nb
            jb = min( nb, min( m, n )-j+1 )
#
#           The active part of the matrix is partitioned
#
#              A11   A12   A13
#              A21   A22   A23
#              A31   A32   A33
#
#           Here A11, A21 and A31 denote the current block of JB columns
#           which is about to be factorized. The number of rows in the
#           partitioning are JB, I2, I3 respectively, and the numbers
#           of columns are JB, J2, J3. The superdiagonal elements of A13
#           and the subdiagonal elements of A31 lie outside the band.
#
            i2 = min( kl-jb, m-j-jb+1 )
            i3 = min( jb, m-j-kl+1 )
#
#           J2 and J3 are computed after JU has been updated.
#
#           Factorize the current block of JB columns
#
            DO 80 jj = j, j + jb - 1
#
#              Set fill-in elements in column JJ+KV to zero
#
               IF( jj+kv.LE.n ) THEN
                  DO 70 i = 1, kl
                     ab( i, jj+kv ) = zero
   70             CONTINUE
               END IF
#
#              Find pivot and test for singularity. KM is the number of
#              subdiagonal elements in the current column.
#
               km = min( kl, m-jj )
               jp = idamax( km+1, ab( kv+1, jj ), 1 )
               ipiv( jj ) = jp + jj - j
               IF( ab( kv+jp, jj ).NE.zero ) THEN
                  ju = max( ju, min( jj+ku+jp-1, n ) )
                  IF( jp.NE.1 ) THEN
#
#                    Apply interchange to columns J to J+JB-1
#
                     IF( jp+jj-1.LT.j+kl ) THEN
#
                        CALL dswap( jb, ab( kv+1+jj-j, j ), ldab-1,
     $                              ab( kv+jp+jj-j, j ), ldab-1 )
                     ELSE
#
#                       The interchange affects columns J to JJ-1 of A31
#                       which are stored in the work array WORK31
#
                        CALL dswap( jj-j, ab( kv+1+jj-j, j ), ldab-1,
     $                              work31( jp+jj-j-kl, 1 ), ldwork )
                        CALL dswap( j+jb-jj, ab( kv+1, jj ), ldab-1,
     $                              ab( kv+jp, jj ), ldab-1 )
                     END IF
                  END IF
#
#                 Compute multipliers
#
                  CALL dscal( km, one / ab( kv+1, jj ), ab( kv+2, jj ),
     $                        1 )
#
#                 Update trailing submatrix within the band and within
#                 the current block. JM is the index of the last column
#                 which needs to be updated.
#
                  jm = min( ju, j+jb-1 )
                  IF( jm.GT.jj )
     $               CALL dger( km, jm-jj, -one, ab( kv+2, jj ), 1,
     $                          ab( kv, jj+1 ), ldab-1,
     $                          ab( kv+1, jj+1 ), ldab-1 )
               ELSE
#
#                 If pivot is zero, set INFO to the index of the pivot
#                 unless a zero pivot has already been found.
#
                  IF( info.EQ.0 )
     $               info = jj
               END IF
#
#              Copy current column of A31 into the work array WORK31
#
               nw = min( jj-j+1, i3 )
               IF( nw.GT.0 )
     $            CALL dcopy( nw, ab( kv+kl+1-jj+j, jj ), 1,
     $                        work31( 1, jj-j+1 ), 1 )
   80       CONTINUE
            IF( j+jb.LE.n ) THEN
#
#              Apply the row interchanges to the other blocks.
#
               j2 = min( ju-j+1, kv ) - jb
               j3 = max( 0, ju-j-kv+1 )
#
#              Use DLASWP to apply the row interchanges to A12, A22, and
#              A32.
#
               CALL dlaswp( j2, ab( kv+1-jb, j+jb ), ldab-1, 1, jb,
     $                      ipiv( j ), 1 )
#
#              Adjust the pivot indices.
#
               DO 90 i = j, j + jb - 1
                  ipiv( i ) = ipiv( i ) + j - 1
   90          CONTINUE
#
#              Apply the row interchanges to A13, A23, and A33
#              columnwise.
#
               k2 = j - 1 + jb + j2
               DO 110 i = 1, j3
                  jj = k2 + i
                  DO 100 ii = j + i - 1, j + jb - 1
                     ip = ipiv( ii )
                     IF( ip.NE.ii ) THEN
                        temp = ab( kv+1+ii-jj, jj )
                        ab( kv+1+ii-jj, jj ) = ab( kv+1+ip-jj, jj )
                        ab( kv+1+ip-jj, jj ) = temp
                     END IF
  100             CONTINUE
  110          CONTINUE
#
#              Update the relevant part of the trailing submatrix
#
               IF( j2.GT.0 ) THEN
#
#                 Update A12
#
                  CALL dtrsm( 'Left', 'Lower', 'No transpose', 'Unit',
     $                        jb, j2, one, ab( kv+1, j ), ldab-1,
     $                        ab( kv+1-jb, j+jb ), ldab-1 )
#
                  IF( i2.GT.0 ) THEN
#
#                    Update A22
#
                     CALL dgemm( 'No transpose', 'No transpose', i2, j2,
     $                           jb, -one, ab( kv+1+jb, j ), ldab-1,
     $                           ab( kv+1-jb, j+jb ), ldab-1, one,
     $                           ab( kv+1, j+jb ), ldab-1 )
                  END IF
#
                  IF( i3.GT.0 ) THEN
#
#                    Update A32
#
                     CALL dgemm( 'No transpose', 'No transpose', i3, j2,
     $                           jb, -one, work31, ldwork,
     $                           ab( kv+1-jb, j+jb ), ldab-1, one,
     $                           ab( kv+kl+1-jb, j+jb ), ldab-1 )
                  END IF
               END IF
#
               IF( j3.GT.0 ) THEN
#
#                 Copy the lower triangle of A13 into the work array
#                 WORK13
#
                  DO 130 jj = 1, j3
                     DO 120 ii = jj, jb
                        work13( ii, jj ) = ab( ii-jj+1, jj+j+kv-1 )
  120                CONTINUE
  130             CONTINUE
#
#                 Update A13 in the work array
#
                  CALL dtrsm( 'Left', 'Lower', 'No transpose', 'Unit',
     $                        jb, j3, one, ab( kv+1, j ), ldab-1,
     $                        work13, ldwork )
#
                  IF( i2.GT.0 ) THEN
#
#                    Update A23
#
                     CALL dgemm( 'No transpose', 'No transpose', i2, j3,
     $                           jb, -one, ab( kv+1+jb, j ), ldab-1,
     $                           work13, ldwork, one, ab( 1+jb, j+kv ),
     $                           ldab-1 )
                  END IF
#
                  IF( i3.GT.0 ) THEN
#
#                    Update A33
#
                     CALL dgemm( 'No transpose', 'No transpose', i3, j3,
     $                           jb, -one, work31, ldwork, work13,
     $                           ldwork, one, ab( 1+kl, j+kv ), ldab-1 )
                  END IF
#
#                 Copy the lower triangle of A13 back into place
#
                  DO 150 jj = 1, j3
                     DO 140 ii = jj, jb
                        ab( ii-jj+1, jj+j+kv-1 ) = work13( ii, jj )
  140                CONTINUE
  150             CONTINUE
               END IF
            ELSE
#
#              Adjust the pivot indices.
#
               DO 160 i = j, j + jb - 1
                  ipiv( i ) = ipiv( i ) + j - 1
  160          CONTINUE
            END IF
#
#           Partially undo the interchanges in the current block to
#           restore the upper triangular form of A31 and copy the upper
#           triangle of A31 back into place
#
            DO 170 jj = j + jb - 1, j, -1
               jp = ipiv( jj ) - jj + 1
               IF( jp.NE.1 ) THEN
#
#                 Apply interchange to columns J to JJ-1
#
                  IF( jp+jj-1.LT.j+kl ) THEN
#
#                    The interchange does not affect A31
#
                     CALL dswap( jj-j, ab( kv+1+jj-j, j ), ldab-1,
     $                           ab( kv+jp+jj-j, j ), ldab-1 )
                  ELSE
#
#                    The interchange does affect A31
#
                     CALL dswap( jj-j, ab( kv+1+jj-j, j ), ldab-1,
     $                           work31( jp+jj-j-kl, 1 ), ldwork )
                  END IF
               END IF
#
#              Copy the current column of A31 back into place
#
               nw = min( i3, jj-j+1 )
               IF( nw.GT.0 )
     $            CALL dcopy( nw, work31( 1, jj-j+1 ), 1,
     $                        ab( kv+kl+1-jj+j, jj ), 1 )
  170       CONTINUE
  180    CONTINUE
      END IF
#
      RETURN
#
#     End of DGBTRF
#
      END
