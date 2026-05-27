#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

struct AtomData { double x,y,z; int type,id; };

struct MaterialCase
{
    std::string name;
    int nx,ny,nz; double lx,ly,lz,cutoff; int ntype,expected_atoms;
    std::vector<std::tuple<int,double,double,double>> basis;
    std::string category;
};

std::vector<AtomData> gen_crystal(const MaterialCase& m)
{
    std::vector<AtomData> a; a.reserve(m.expected_atoms);
    int id=0;
    for(int ix=0;ix<m.nx;++ix) for(int iy=0;iy<m.ny;++iy) for(int iz=0;iz<m.nz;++iz)
        for(auto& b:m.basis) {
            if(id>=m.expected_atoms) break;
            a.push_back({(ix+std::get<1>(b))*m.lx,(iy+std::get<2>(b))*m.ly,(iz+std::get<3>(b))*m.lz,
                         std::get<0>(b),id++});
        }
    return a;
}

std::vector<AtomData> gen_cluster(int n,double box,double frac)
{
    std::vector<AtomData> a(n);
    int na=(int)(n*frac);
    double cx=box*.3,cy=box*.3,cz=box*.3,cr=box*.07;
    std::srand(12345);
    for(int i=0;i<na;i++){
        double r=cr*std::pow(rand()/(double)RAND_MAX,1./3.);
        double th=6.283185307*rand()/RAND_MAX;
        double ph=acos(2.*rand()/RAND_MAX-1.);
        a[i]={cx+r*sin(ph)*cos(th),cy+r*sin(ph)*sin(th),cz+r*cos(ph),0,i};
    }
    for(int i=na;i<n;i++)
        a[i]={box*rand()/RAND_MAX,box*rand()/RAND_MAX,box*rand()/RAND_MAX,0,i};
    return a;
}

struct Grid {
    int nx,ny,nz; double x0,y0,z0;
    std::vector<std::vector<std::vector<std::vector<int>>>> g;
};
Grid build_grid(const std::vector<AtomData>& a,double gs)
{
    Grid gr; int n=(int)a.size();
    gr.x0=gr.y0=gr.z0=a[0].x;
    double xm=a[0].x,ym=a[0].y,zm=a[0].z;
    for(auto& t:a){gr.x0=std::min(gr.x0,t.x);xm=std::max(xm,t.x);
                   gr.y0=std::min(gr.y0,t.y);ym=std::max(ym,t.y);
                   gr.z0=std::min(gr.z0,t.z);zm=std::max(zm,t.z);}
    gr.nx=(int)ceil((xm-gr.x0)/gs)+1; gr.ny=(int)ceil((ym-gr.y0)/gs)+1; gr.nz=(int)ceil((zm-gr.z0)/gs)+1;
    gr.g.assign(gr.nx,std::vector<std::vector<std::vector<int>>>(gr.ny,std::vector<std::vector<int>>(gr.nz)));
    for(int i=0;i<n;i++){
        int bx=std::max(0,std::min(gr.nx-1,(int)floor((a[i].x-gr.x0)/gs)));
        int by=std::max(0,std::min(gr.ny-1,(int)floor((a[i].y-gr.y0)/gs)));
        int bz=std::max(0,std::min(gr.nz-1,(int)floor((a[i].z-gr.z0)/gs)));
        gr.g[bx][by][bz].push_back(i);
    }
    return gr;
}

std::vector<double> atom_weights(const std::vector<AtomData>& a,double cut,double gs,const Grid& gr)
{
    int n=(int)a.size(),sr=(int)ceil(cut/gs); std::vector<double> w(n,1.);
    for(int i=0;i<n;i++){
        int bx=std::max(0,std::min(gr.nx-1,(int)floor((a[i].x-gr.x0)/gs)));
        int by=std::max(0,std::min(gr.ny-1,(int)floor((a[i].y-gr.y0)/gs)));
        int bz=std::max(0,std::min(gr.nz-1,(int)floor((a[i].z-gr.z0)/gs)));
        int nb=0;
        for(int dx=-sr;dx<=sr;dx++){int cx=bx+dx;if(cx<0||cx>=gr.nx)continue;
            for(int dy=-sr;dy<=sr;dy++){int cy=by+dy;if(cy<0||cy>=gr.ny)continue;
                for(int dz=-sr;dz<=sr;dz++){int cz=bz+dz;if(cz<0||cz>=gr.nz)continue;
                    nb+=(int)gr.g[cx][cy][cz].size();}}}
        w[i]=std::max(1.,(double)nb);
    }
    return w;
}

struct Decomp { std::vector<int> off; double imb; };
Decomp distribute(int n,const std::vector<double>& w,int np)
{
    std::vector<double> p(n+1,0.); for(int i=0;i<n;i++) p[i+1]=p[i]+w[i];
    double tot=p[n],per=tot/np; Decomp d; d.off.resize(np+1,0); d.off[np]=n;
    int cur=0;
    for(int r=1;r<np;r++){double t=r*per;while(cur<n&&p[cur+1]<t)cur++;d.off[r]=std::min(n,cur+1);}
    double mx=0,mn=1e100;
    for(int r=0;r<np;r++){double ld=p[d.off[r+1]]-p[d.off[r]];mx=std::max(mx,ld);mn=std::min(mn,ld);}
    d.imb = (mn > 0) ? mx / mn : 1.0;
    return d;
}

static void search_atoms(const std::vector<AtomData>& a, int start, int end, double cut2, double gs,
                          const Grid& gr, double& cnt)
{
    double lc = 0;
    int sr = static_cast<int>(std::ceil(std::sqrt(cut2 / (gs * gs))));
    for (int i = start; i < end; i++)
    {
        int bx = std::max(0, std::min(gr.nx - 1, static_cast<int>(std::floor((a[i].x - gr.x0) / gs))));
        int by = std::max(0, std::min(gr.ny - 1, static_cast<int>(std::floor((a[i].y - gr.y0) / gs))));
        int bz = std::max(0, std::min(gr.nz - 1, static_cast<int>(std::floor((a[i].z - gr.z0) / gs))));
        for (int dx = -sr; dx <= sr; dx++)
        {
            int cx = bx + dx;
            if (cx < 0 || cx >= gr.nx)
                continue;
            for (int dy = -sr; dy <= sr; dy++)
            {
                int cy = by + dy;
                if (cy < 0 || cy >= gr.ny)
                    continue;
                for (int dz = -sr; dz <= sr; dz++)
                {
                    int cz = bz + dz;
                    if (cz < 0 || cz >= gr.nz)
                        continue;
                    for (int j : gr.g[cx][cy][cz])
                    {
                        if (j <= i)
                            continue;
                        double dx2 = a[i].x - a[j].x;
                        double dy2 = a[i].y - a[j].y;
                        double dz2 = a[i].z - a[j].z;
                        if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < cut2)
                            lc += 1.0;
                    }
                }
            }
        }
    }
    cnt += lc;
}

double run_serial(const std::vector<AtomData>& a,double cut,int& nb)
{
    double t0=omp_get_wtime();
    double gs=cut+.1,c2=cut*cut; Grid gr=build_grid(a,gs);
    double cnt=0; search_atoms(a,0,(int)a.size(),c2,gs,gr,cnt);
    nb=(int)cnt; return (omp_get_wtime()-t0)*1000.;
}

double run_static(const std::vector<AtomData>& a,double cut,MPI_Comm comm,int& nb,double& imb,double& tmax,double& tmin)
{
    int rk,np; MPI_Comm_rank(comm,&rk); MPI_Comm_size(comm,&np);
    double t0=MPI_Wtime();
    int n=(int)a.size(); double gs=cut+.1,c2=cut*cut;

    int base=n/np,rem=n%np;
    int start=rk*base+std::min(rk,rem), end=start+base+(rk<rem?1:0);

    Grid gr=build_grid(a,gs);

    double tc0=MPI_Wtime(),cnt=0;
#pragma omp parallel
    {
        double lc = 0;
        int sr = static_cast<int>(std::ceil(std::sqrt(c2 / (gs * gs))));
#pragma omp for schedule(static) nowait
        for (int i = start; i < end; i++)
        {
            int bx = std::max(0, std::min(gr.nx - 1, static_cast<int>(std::floor((a[i].x - gr.x0) / gs))));
            int by = std::max(0, std::min(gr.ny - 1, static_cast<int>(std::floor((a[i].y - gr.y0) / gs))));
            int bz = std::max(0, std::min(gr.nz - 1, static_cast<int>(std::floor((a[i].z - gr.z0) / gs))));
            for (int dx = -sr; dx <= sr; dx++)
            {
                int cx = bx + dx;
                if (cx < 0 || cx >= gr.nx)
                    continue;
                for (int dy = -sr; dy <= sr; dy++)
                {
                    int cy = by + dy;
                    if (cy < 0 || cy >= gr.ny)
                        continue;
                    for (int dz = -sr; dz <= sr; dz++)
                    {
                        int cz = bz + dz;
                        if (cz < 0 || cz >= gr.nz)
                            continue;
                        for (int j : gr.g[cx][cy][cz])
                        {
                            if (j <= i)
                                continue;
                            double dx2 = a[i].x - a[j].x;
                            double dy2 = a[i].y - a[j].y;
                            double dz2 = a[i].z - a[j].z;
                            if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < c2)
                                lc += 1.0;
                        }
                    }
                }
            }
        }
#pragma omp atomic
        cnt += lc;
    }
    double tcomp = MPI_Wtime() - tc0;

    int my=(int)cnt; MPI_Allreduce(&my,&nb,1,MPI_INT,MPI_SUM,comm);

    std::vector<int> allc(np); int lc=end-start;
    MPI_Allgather(&lc,1,MPI_INT,allc.data(),1,MPI_INT,comm);
    int mx=*std::max_element(allc.begin(),allc.end()),mn=*std::min_element(allc.begin(),allc.end());
    imb=(mn>0)?(double)mx/mn:1.;

    std::vector<double> allt(np);
    MPI_Gather(&tcomp,1,MPI_DOUBLE,allt.data(),1,MPI_DOUBLE,0,comm);
    if(rk==0){tmax=*std::max_element(allt.begin(),allt.end());tmin=*std::min_element(allt.begin(),allt.end());}

    return (MPI_Wtime()-t0)*1000.;
}

double run_dynamic(const std::vector<AtomData>& a,double cut,MPI_Comm comm,int& nb,double& imb,double& tmax,double& tmin)
{
    int rk,np; MPI_Comm_rank(comm,&rk); MPI_Comm_size(comm,&np);
    double t0=MPI_Wtime();
    int n=(int)a.size(); double gs=cut+.1,c2=cut*cut;

    Grid gr=build_grid(a,gs);
    std::vector<double> w=atom_weights(a,cut,gs,gr);
    Decomp d=distribute(n,w,np);

    MPI_Bcast(d.off.data(),np+1,MPI_INT,0,comm);
    MPI_Bcast(&d.imb,1,MPI_DOUBLE,0,comm);

    int start=d.off[rk],end=d.off[rk+1];

    double tc0=MPI_Wtime(),cnt=0;
#pragma omp parallel
    {
        double lc = 0;
        int sr = static_cast<int>(std::ceil(std::sqrt(c2 / (gs * gs))));
#pragma omp for schedule(static) nowait
        for (int i = start; i < end; i++)
        {
            int bx = std::max(0, std::min(gr.nx - 1, static_cast<int>(std::floor((a[i].x - gr.x0) / gs))));
            int by = std::max(0, std::min(gr.ny - 1, static_cast<int>(std::floor((a[i].y - gr.y0) / gs))));
            int bz = std::max(0, std::min(gr.nz - 1, static_cast<int>(std::floor((a[i].z - gr.z0) / gs))));
            for (int dx = -sr; dx <= sr; dx++)
            {
                int cx = bx + dx;
                if (cx < 0 || cx >= gr.nx)
                    continue;
                for (int dy = -sr; dy <= sr; dy++)
                {
                    int cy = by + dy;
                    if (cy < 0 || cy >= gr.ny)
                        continue;
                    for (int dz = -sr; dz <= sr; dz++)
                    {
                        int cz = bz + dz;
                        if (cz < 0 || cz >= gr.nz)
                            continue;
                        for (int j : gr.g[cx][cy][cz])
                        {
                            if (j <= i)
                                continue;
                            double dx2 = a[i].x - a[j].x;
                            double dy2 = a[i].y - a[j].y;
                            double dz2 = a[i].z - a[j].z;
                            if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < c2)
                                lc += 1.0;
                        }
                    }
                }
            }
        }
#pragma omp atomic
        cnt += lc;
    }
    double tcomp = MPI_Wtime() - tc0;

    int my=(int)cnt; MPI_Allreduce(&my,&nb,1,MPI_INT,MPI_SUM,comm);
    std::vector<double> allt(np);
    MPI_Gather(&tcomp,1,MPI_DOUBLE,allt.data(),1,MPI_DOUBLE,0,comm);
    if(rk==0){imb=d.imb; tmax=*std::max_element(allt.begin(),allt.end());tmin=*std::min_element(allt.begin(),allt.end());}

    return (MPI_Wtime()-t0)*1000.;
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    int rk,np; MPI_Comm_rank(MPI_COMM_WORLD,&rk); MPI_Comm_size(MPI_COMM_WORLD,&np);
    int nt=1;
#pragma omp parallel
#pragma omp master
    nt=omp_get_num_threads();

    struct MM {std::string name;int nx,ny,nz;double lx,ly,lz,cut;int nt,nat; std::vector<std::tuple<int,double,double,double>> bs;std::string cat;};
    std::vector<MM> mats={
        {"Al fcc",10,5,5,2,2,2,1.5,1,1000,{{0,0,0,0},{0,0,.5,.5},{0,.5,0,.5},{0,.5,.5,0}},"uniform"},
        {"Si diamond",10,5,5,2.4,2.4,2.4,1.25,1,2000,
         {{0,0,0,0},{0,0,.5,.5},{0,.5,0,.5},{0,.5,.5,0},{0,.25,.25,.25},{0,.25,.75,.75},{0,.75,.25,.75},{0,.75,.75,.25}},"uniform"},
        {"NaCl",15,5,5,2.4,2.4,2.4,1.35,2,3000,
         {{0,0,0,0},{0,0,.5,.5},{0,.5,0,.5},{0,.5,.5,0},{1,.5,0,0},{1,0,.5,0},{1,0,0,.5},{1,.5,.5,.5}},"uniform"},
        {"TiO2 rutile",10,10,7,2.4,2.4,1.56,1.25,2,4200,
         {{0,0,0,0},{0,.5,.5,.5},{1,.305,.305,0},{1,.695,.695,0},{1,.805,.195,.5},{1,.195,.805,.5}},"uniform"},
    };

    if(rk==0){std::cout<<"\n=========================================================\n";
        std::cout<<"  ABACUS Neighbor Search - Low Overhead MPI+OpenMP\n";
        std::cout<<"  Cores: "<<np*nt<<" ("<<np<<" MPI x "<<nt<<" OMP)\n";
        std::cout<<"  Opt: no-atom-Bcast, packed params, Allreduce, seed-gen\n";
        std::cout<<"=========================================================\n";}

    struct R {std::string nm;int na;double ser,st,ss,se,si,sx,sn,dt,ds,de,di,dx,dn;};
    std::vector<R> rs;

    for(size_t mi=0;mi<mats.size();mi++){
        auto& m=mats[mi];
        std::vector<AtomData> at=gen_crystal({m.name,m.nx,m.ny,m.nz,m.lx,m.ly,m.lz,m.cut,m.nt,m.nat,m.bs,""});

        double ser=0; int snb=0; if(rk==0) ser=run_serial(at,m.cut,snb);

        double st,sim,stx,stn; int stnb=0;
        st=run_static(at,m.cut,MPI_COMM_WORLD,stnb,sim,stx,stn);

        double dt,dim,dtx,dtn; int dtnb=0;
        dt=run_dynamic(at,m.cut,MPI_COMM_WORLD,dtnb,dim,dtx,dtn);

        if(rk==0){double u=np*nt;
            rs.push_back({m.name,(int)at.size(),ser,st,ser/st,ser/st/u*100,sim,stx*1000,stn*1000,dt,ser/dt,ser/dt/u*100,dim,dtx*1000,dtn*1000});}
    }
    {
        double cut=3.0;
        std::vector<AtomData> at=gen_cluster(5000,100.,.7);

        double ser=0; int snb=0; if(rk==0) ser=run_serial(at,cut,snb);

        double st,sim,stx,stn; int stnb=0;
        st=run_static(at,cut,MPI_COMM_WORLD,stnb,sim,stx,stn);

        double dt,dim,dtx,dtn; int dtnb=0;
        dt=run_dynamic(at,cut,MPI_COMM_WORLD,dtnb,dim,dtx,dtn);

        if(rk==0){double u=np*nt;
            rs.push_back({"Cluster70%",5000,ser,st,ser/st,ser/st/u*100,sim,stx*1000,stn*1000,dt,ser/dt,ser/dt/u*100,dim,dtx*1000,dtn*1000});}
    }

    if(rk==0){
        std::cout<<"\n  "<<std::left<<std::setw(16)<<"Case"<<std::right<<std::setw(6)<<"Atoms"
                  <<std::setw(10)<<"Serial"<<std::setw(10)<<"Static"<<std::setw(6)<<"SU"<<std::setw(7)<<"Eff%"
                  <<std::setw(8)<<"TmaxS"<<std::setw(8)<<"TminS"
                  <<std::setw(10)<<"Dynamic"<<std::setw(6)<<"SU"<<std::setw(7)<<"Eff%"
                  <<std::setw(8)<<"TmaxD"<<std::setw(8)<<"TminD"<<"\n  "<<std::string(110,'-')<<"\n";
        double ts=0,tsta=0,tdyn=0;
        for(auto& r:rs){ts+=r.ser;tsta+=r.st;tdyn+=r.dt;
            std::cout<<"  "<<std::left<<std::setw(16)<<r.nm<<std::right<<std::setw(6)<<r.na
                      <<std::setw(10)<<std::fixed<<std::setprecision(1)<<r.ser
                      <<std::setw(10)<<std::fixed<<std::setprecision(1)<<r.st
                      <<std::setw(5)<<std::fixed<<std::setprecision(1)<<r.ss<<"x"
                      <<std::setw(6)<<std::fixed<<std::setprecision(1)<<r.se<<"%"
                      <<std::setw(8)<<std::fixed<<std::setprecision(1)<<r.sx
                      <<std::setw(8)<<std::fixed<<std::setprecision(1)<<r.sn
                      <<std::setw(10)<<std::fixed<<std::setprecision(1)<<r.dt
                      <<std::setw(5)<<std::fixed<<std::setprecision(1)<<r.ds<<"x"
                      <<std::setw(6)<<std::fixed<<std::setprecision(1)<<r.de<<"%"
                      <<std::setw(8)<<std::fixed<<std::setprecision(1)<<r.dx
                      <<std::setw(8)<<std::fixed<<std::setprecision(1)<<r.dn<<"\n";}
        std::cout<<"  "<<std::string(110,'-')<<"\n";
        double u=np*nt,os=ts/tsta,od=ts/tdyn;
        std::cout<<"  "<<std::left<<std::setw(16)<<"OVERALL"<<std::right<<std::setw(6)<<""
                  <<std::setw(10)<<std::fixed<<std::setprecision(1)<<ts<<std::setw(10)<<tsta
                  <<std::setw(5)<<os<<"x"<<std::setw(6)<<(os/u*100)<<"%"
                  <<std::setw(8)<<"-"<<std::setw(8)<<"-"<<std::setw(10)<<tdyn<<std::setw(5)<<od<<"x"
                  <<std::setw(6)<<(od/u*100)<<"%\n";
        std::cout<<"  Static: "<<std::fixed<<std::setprecision(2)<<os<<"x, "<<(os/u*100)<<"% eff\n";
        std::cout<<"  Dynamic:"<<std::fixed<<std::setprecision(2)<<od<<"x, "<<(od/u*100)<<"% eff\n";
        std::cout<<"=========================================================\n";
    }
    MPI_Finalize(); return 0;
}
