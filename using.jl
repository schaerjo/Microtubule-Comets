using Pkg

@inline function using_pkg(st)
    listofpkg = split(st, ", ")
    for package in listofpkg
        try 
            eval(Meta.parse(string("using ", package)))
            # @eval using $(Symbol(package))
        catch
            println("Installing $package ...")
            Pkg.add(package)
            # @eval using $(Symbol(package))
            eval(Meta.parse(string("using ", package)))
        end
    end
end

@inline function using_mod(st)
    listofmod = split(st, ", ")
    for modul in listofmod
        try 
            eval(Meta.parse(string("using ", modul)))
        catch
            include(string(@__DIR__,"/",modul[2:end],".jl"))
            eval(Meta.parse(string("using ", modul)))
        end
    end
end

@inline function import_pkg(st)
    listofpkg = split(st, ", ")
    for package in listofpkg
        try 
            eval(Meta.parse(string("import ", package)))
        catch
            println("Installing $package ...")
            Pkg.add(package)
            eval(Meta.parse(string("import ", package)))
        end
    end
end

# if !isdefined(@__MODULE__, :PictUtils)
#     include("../Utilities/png_utilities.jl")
# end
